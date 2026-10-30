"""Trainer class for AtomTrick."""

import json
import logging
import os
import time
from datetime import timedelta

from tqdm import tqdm

import torch
import torch.nn as nn

from .model import AtomTrick
from .data import MLFFDataset, build_dataloader

logger = logging.getLogger(__name__)


def collect_provenance(device=None):
    """Code / software / hardware provenance for ``args.json`` (hostname-free).

    ``git_sha`` / ``git_dirty`` come from the git checkout that contains this
    package (``None`` when it is not a git checkout or git is unavailable).
    """
    import platform
    import subprocess
    from datetime import datetime, timezone

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _git(*args):
        try:
            out = subprocess.run(["git", *args], cwd=root, capture_output=True,
                                 text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no") if sha else None
    cuda_device = None
    try:
        dev = torch.device(device) if device is not None else None
        if dev is not None and dev.type == "cuda" and torch.cuda.is_available():
            cuda_device = torch.cuda.get_device_name(dev)
    except Exception:           # noqa: BLE001 -- provenance must never fail a run
        cuda_device = None
    return {
        "git_sha": sha,
        "git_dirty": (bool(status) if status is not None else None),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device": cuda_device,
        "python_version": platform.python_version(),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def check_cutoffs(config):
    """Raise ``ValueError`` unless ``edge_cutoff`` equals ``cutoff``.

    ``edge_cutoff`` is the neighbour-list radius (``MLFFDataset``, read by
    :meth:`Trainer.build_loaders` and ``evaluate.py``); ``cutoff`` is where the
    radial basis -- and with it every attention / embedding gate -- goes to zero
    (``AtomTrick(cutoff=...)``).  Messages vanish smoothly at the cutoff only
    when the two agree.  With ``edge_cutoff < cutoff`` edges are dropped while
    their gate is still non-zero, so the energy jumps whenever an atom crosses
    ``edge_cutoff``.  With ``edge_cutoff > cutoff`` the extra edges carry zero
    messages in the paper model, but they still enter softmax normalisation
    (``attn_nonlin="softmax"``), i.e. a different model.  Either
    way the run would silently differ from the one described, so it is
    rejected here.

    The defaults are the ones the consumers apply to a missing key (5.0 for
    both), so a config that sets only ``cutoff`` to something other than 5.0 is
    rejected too: set ``edge_cutoff`` to the same value.  An explicit
    ``edge_cutoff: null`` means the graph comes from the NPZ's stored
    ``edge_index`` and is not checked.  Every config and run of the paper
    has ``cutoff == edge_cutoff == 5.0``.
    """
    cutoff = config.get("cutoff", 5.0)
    edge_cutoff = config.get("edge_cutoff", 5.0)
    if cutoff is None or edge_cutoff is None:
        return
    if abs(float(cutoff) - float(edge_cutoff)) > 1e-9:
        raise ValueError(
            f"edge_cutoff ({float(edge_cutoff)} A, neighbour-list radius) must equal "
            f"cutoff ({float(cutoff)} A, radial-basis cutoff); a missing key defaults "
            f"to 5.0.  Set both to the same value (e.g. --cutoff X --edge_cutoff X).")


def inference_cutoff(config, override=None):
    """Neighbour-list radius for graphs built at inference (MD, benchmarks).

    Returns the model's radial-basis ``cutoff`` (default 5.0, as in
    :meth:`Trainer.build_model`), which :func:`check_cutoffs` keeps equal
    to the training ``edge_cutoff`` whenever that is set.  ``override`` (a
    ``--cutoff`` flag) changes only the neighbour list, not the radial basis,
    so it must match: any other value raises ``ValueError`` for the reasons
    given in :func:`check_cutoffs`.
    """
    cutoff = float(config.get("cutoff", 5.0))
    if override is not None and abs(float(override) - cutoff) > 1e-9:
        raise ValueError(
            f"--cutoff {float(override)} A (neighbour list) differs from the model's "
            f"radial-basis cutoff ({cutoff} A), i.e. a different model from the one "
            f"trained (see check_cutoffs).  Omit --cutoff to use the model's value.")
    return cutoff


class Trainer:
    """Manages the full training lifecycle for AtomTrick.

    Parameters
    ----------
    model : AtomTrick
    device : torch.device
    out_dir : str
    rho_force : float
        Weight for force loss; energy weight is ``1 - rho_force``.
    grad_clip : float
    save_every : int
        Save a checkpoint every this many iterations.
    eval_every : int
        Run val/test evaluation every this many iterations.
    lr : float
        Learning rate for AdamW.
    weight_decay : float
        Weight decay for AdamW.
    scheduler_patience : int
        Patience (eval rounds) for ReduceLROnPlateau.
    scheduler_factor : float
        Multiplicative LR reduction factor for ReduceLROnPlateau.
    scheduler_min_lr : float
        Minimum LR for ReduceLROnPlateau.
    warmup_steps : int
        Number of steps for linear LR warmup from 0 to ``lr``.
        Set to 0 to disable warmup.
    wandb_project : str | None
        Weights & Biases project name.  ``None`` disables wandb logging.
    wandb_name : str | None
        Run name for wandb.
    """

    def __init__(self, model, *, device=None,
                 out_dir="checkpoints", rho_force=0.5,
                 grad_clip=10.0, save_every=1000, eval_every=500,
                 lr=1e-3, weight_decay=1e-5,
                 scheduler_patience=10, scheduler_factor=0.5,
                 scheduler_min_lr=1e-6, warmup_steps=1000,
                 wandb_project=None, wandb_name=None,
                 ema_decay=0.0,
                 swa_start=0, swa_rho_force=0.1, swa_lr=0.0,
                 optimizer="adamw"):
        self.model = model
        self.device = device or torch.device("cpu")
        self.out_dir = out_dir

        # ---- Distributed data parallel (auto-detected from torchrun/srun env) --
        # WORLD_SIZE>1 => run under DDP: each rank owns one GPU and a data shard;
        # gradients are all-reduced by torch's DistributedDataParallel (the make_fx
        # single-backward makes this the ordinary supported case).  Rank 0 alone
        # does eval / logging / checkpoints; the plateau scheduler steps on a
        # broadcast metric so every rank keeps an identical LR (and Adam state).
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.is_dist = self.world_size > 1
        self.is_main = (self.rank == 0)
        # GPU this rank uses; local_rank % device_count so a 1-GPU dev box can
        # host 2 ranks (gloo smoke) while real 2-GPU maps 0->0, 1->1.
        _ndev = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.gpu_index = self.local_rank % max(1, _ndev)
        self._ddp_module = None       # lazily-built DDP wrapper (make_fx path)
        if self.is_dist:
            logger.info("DDP: rank %d/%d local_rank %d device %s",
                        self.rank, self.world_size, self.local_rank, self.device)
        self.rho_force = rho_force
        self.grad_clip = grad_clip
        self.save_every = save_every
        self.eval_every = eval_every

        # ---- Optimizer -------------------------------------------------------
        # AdamW over all parameters in one group (the only optimizer).
        self.optimizer_name = str(optimizer).lower()
        if self.optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"unknown optimizer {optimizer!r} (only 'adamw' is supported)")
        self.warmup_steps = warmup_steps
        if warmup_steps > 0:
            self._warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1e-8, end_factor=1.0,
                total_iters=warmup_steps,
            )
        else:
            self._warmup_scheduler = None

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr,
        )

        # Store scheduler config so a scheduler can be re-created on the
        # SWA stage switch.
        self._base_lr = lr
        self._weight_decay = weight_decay
        self._scheduler_cfg = dict(
            factor=scheduler_factor, patience=scheduler_patience,
            min_lr=scheduler_min_lr,
        )

        # ---- Training step selection ----------------------------------------
        # Two paths, both plain autograd over F = -dE/dR:
        #   * ATOMTRICK_MAKEFX=1 — make_fx bakes the forward + the force
        #     1st-backward into ONE aten graph, so the training step is a single
        #     loss.backward() that torch.compile can handle, dynamic across all
        #     batch shapes (one graph, no recompile storm).  The production path.
        #     Needs PYTORCH_JIT=0 in the launch env (TorchScript breaks make_fx's
        #     2nd-order trace).
        #   * otherwise — eager autograd: model.forward runs
        #     autograd.grad(E, R, create_graph=True), loss.backward() supplies the
        #     parameter grads.  Used for eval/inference and as the fallback.
        self.use_makefx = os.environ.get("ATOMTRICK_MAKEFX", "0") == "1"
        self._makefx_step = None
        self._makefx_infer = None     # compiled INFERENCE graph; built lazily on
                                      # 1st eval when no training graph exists
        self._logged_runt_pad = False  # log the runt-batch padding once
        if self.use_makefx:
            logger.info("ATOMTRICK_MAKEFX=1: fully-dynamic make_fx + torch.compile "
                        "training step (needs PYTORCH_JIT=0).")

        self.best_val_loss = float("inf")
        self._no_improve_count = 0
        self.step = 0
        self._wandb_name = wandb_name

        # ---- EMA (Exponential Moving Average) --------------------------------
        self.ema_decay = ema_decay
        if ema_decay > 0:
            self._ema_model = torch.optim.swa_utils.AveragedModel(
                model,
                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay),
            )
            logger.info("EMA enabled with decay=%.6f", ema_decay)
        else:
            self._ema_model = None

        # ---- MACE-style two-stage SWA (opt-in via swa_start > 0) -------------
        self.swa_start = swa_start
        self.swa_rho_force = swa_rho_force
        self.swa_lr = swa_lr if swa_lr > 0 else lr
        self._swa_entered = False

        os.makedirs(self.out_dir, exist_ok=True)

        # ---- wandb -----------------------------------------------------------
        self._wandb = None
        if wandb_project is not None and self.is_main:
            import wandb
            self._wandb = wandb
            wandb.init(project=wandb_project, name=wandb_name,
                       dir=self.out_dir, resume="allow")

    # ------------------------------------------------------------------
    # Loss (used for backward)
    # ------------------------------------------------------------------

    @staticmethod
    def combined_loss(pred, batch, rho_force):
        loss_F = (pred["F"] - batch["F"]).pow(2).mean()
        loss_E = (pred["E"] - batch["E"]).pow(2).mean()
        return rho_force * loss_F + (1 - rho_force) * loss_E, loss_F.detach(), loss_E.detach()

    def _in_swa_stage(self, step):
        """True if a trainer whose step counter is ``step`` has already entered
        the SWA stage.  The switch fires at the top of the iteration that sees
        ``self.step >= swa_start``, i.e. AFTER a checkpoint saved at
        ``step == swa_start`` was written, so only ``step > swa_start`` is inside
        the stage."""
        return self.swa_start > 0 and step > self.swa_start

    def _enter_swa_stage(self):
        """MACE-style two-stage switch (fires once when self.step reaches swa_start).

        Flips the loss to energy-heavy (rho_force -> swa_rho_force), resets the LR
        (the plateau scheduler will have decayed it near the floor by now, leaving
        no room to retrain the energy head) and the plateau scheduler, and restarts
        weight-averaging as a fresh UNIFORM SWA average so stage-2 (energy-tuned)
        weights are not diluted by stage-1's force-tuned ones.
        """
        self.rho_force = self.swa_rho_force
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.swa_lr
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", **self._scheduler_cfg)
        if self._ema_model is not None:
            self._ema_model = torch.optim.swa_utils.AveragedModel(self.model)
        # Restart best-val tracking for the SWA (energy) stage so it saves its OWN best.pt.
        # The carried-over best_val_loss is on the force-heavy (rho 0.999) scale, which the
        # energy-heavy (rho 0.1) SWA val loss never beats -> best.pt would otherwise never fire,
        # and the uniform SWA average drifts past its early energy peak (exp2 100k-from-600k
        # lost a ~0.44 meV/at peak this way). Reset so best.pt tracks the SWA-phase minimum.
        self.best_val_loss = float("inf")
        self._no_improve_count = 0
        self._swa_entered = True
        logger.info("SWA stage @ step %d: rho_force -> %.4f, lr -> %.2e, "
                    "plateau scheduler + uniform SWA average reset",
                    self.step, self.rho_force, self.swa_lr)

    # ------------------------------------------------------------------
    # Metrics: energy MAE, force mean-L2, force MAE (pooled sums)
    # ------------------------------------------------------------------

    def _compute_metrics(self, pred, batch):
        """Per-batch error SUMS, for pooling over a whole data set.

        Applies the model's ``energy_bias`` buffer (always zero) to
        ``pred["E"]`` before comparing against ``batch["E"]``, as
        ``add_atomic_reference`` does at inference.  The errors are accumulated
        in float64.

        Pool as ``sum(e_abs) / sum(n_mol)``, ``sum(f_l2) / sum(n_atom)`` and
        ``sum(f_abs) / (3 * sum(n_atom))``: pooled, not an average of
        per-batch means, which would over-weight a smaller final batch.

        Returns
        -------
        e_abs : float
            Sum over molecules of |E_pred - E_ref| (eV).
        f_l2 : float
            Sum over atoms of the L2 norm of the force error (eV/Å).
        f_abs : float
            Sum over atoms × 3 Cartesian components of |F_pred - F_ref| (eV/Å).
        n_mol : int
            Number of molecules in the batch.
        n_atom : int
            Number of atoms in the batch.
        """
        with torch.no_grad():
            f_err = pred["F"].double() - batch["F"].double()
            n_mol = batch["E"].shape[0]
            bias = self.model.energy_bias_per_mol(
                batch["Z"], batch["batch"], n_mol,
            ).double()
            e_abs = (pred["E"].double() + bias - batch["E"].double()).abs().sum().item()
            f_l2 = f_err.norm(dim=-1).sum().item()
            f_abs = f_err.abs().sum().item()
        n_atom = batch["F"].shape[0]
        return e_abs, f_l2, f_abs, n_mol, n_atom

    # ------------------------------------------------------------------
    # EMA helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _ema_update(self):
        """Update EMA shadow weights via AveragedModel."""
        if self._ema_model is None:
            return
        self._ema_model.update_parameters(self.model)

    def _ema_apply(self):
        """Temporarily swap self.model to the EMA model for evaluation."""
        if self._ema_model is None:
            return
        self._orig_model = self.model
        self.model = self._ema_model.module

    def _ema_restore(self):
        """Restore the original (training) model after EMA evaluation."""
        if self._ema_model is None:
            return
        self.model = self._orig_model
        del self._orig_model

    # ------------------------------------------------------------------
    # Distributed helper
    # ------------------------------------------------------------------

    def _bcast_scalar(self, value, src=0):
        """Broadcast a python float from ``src`` to all ranks and return it.

        Used to keep the plateau scheduler (and the early-stop decision) in sync:
        rank 0 computes the val metric, every rank steps the scheduler with the
        SAME number so their LR / optimizer state never diverge.  No-op (returns
        ``value``) when not running distributed.  The tensor is freshly built
        (not a compiled-graph output) so ``.item()`` carries no symbolic tag.
        """
        if not self.is_dist:
            return value
        import torch.distributed as dist
        t = torch.tensor([float(value)], device=self.device)
        dist.broadcast(t, src=src)
        return t.item()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _forward(self, batch):
        """Run the model forward (eval / predict) → ``{"E", "F"}``.

        Eval/inference forces come from the eager autograd path
        (``model.forward`` = ``_energy`` + ``autograd.grad(E, R)``), the SAME
        path make_fx traces for training.

        make_fx path: uses the SAME make_fx + torch.compile machinery as training —
        a dedicated inference graph traced with ``create_graph=True`` (the proven
        bit-exact force trace; ``create_graph=False`` mis-traces under make_fx →
        wrong forces).  We just never call ``loss.backward()`` on it and read/detach
        E,F.  One dynamic graph serves every batch shape; falls back to eager
        ``model(batch)`` if the compile fails.
        """
        if self.use_makefx:
            return self._makefx_infer_forward(batch)
        return self.model(batch)

    def _makefx_infer_forward(self, batch):
        """Compiled inference (make_fx + torch.compile).

        Runs the make_fx graph under no_grad and detaches/plainifies E,F so
        downstream ``.item()`` / metrics stay safe.  The graph is traced with
        ``create_graph=True`` (the SAME trace training uses — the only one make_fx
        captures correctly; the leaner ``create_graph=False`` eval trace produces
        WRONG forces).  We REUSE the already-built training graph
        (``self._makefx_step``) when it exists — no second compile, and no
        2nd-order build memory.  Only PURE inference (no training happened) builds
        a dedicated graph, once, cached.  Any trace/compile failure logs once and
        permanently falls back to eager ``model(batch)``.
        """
        from .backward.makefx_compiled import MakeFXCompiledStep, plainify
        step = self._makefx_step
        if step is None:
            if self._makefx_infer is None:
                try:
                    # tracing the force autograd.grad needs grad even though eval
                    # wraps this in no_grad (model.forward also enable_grads).
                    with torch.enable_grad():
                        self._makefx_infer = MakeFXCompiledStep(
                            self.model, batch, dynamic=True)
                    logger.info("make_fx INFERENCE step compiled: %d graph nodes, "
                                "%d complex ops, %d detach/alias stripped",
                                self._makefx_infer._n_nodes,
                                self._makefx_infer._n_complex,
                                self._makefx_infer._n_detach)
                except Exception as e:      # noqa: BLE001 — any trace/compile failure
                    logger.warning("make_fx inference compile failed (%s); using "
                                   "eager model(batch) for eval/inference.", e)
                    self._makefx_infer = False   # sentinel: don't retry
            step = self._makefx_infer
        if step:
            # The compiled graph's per-molecule energy output is specialised to
            # the molecule count baked at trace time (step._traced_nmol = the
            # train/first-eval batch_size): make_fx makes atom/edge counts
            # dynamic, but n_mol is CONSTANT during training (drop_last keeps
            # every train batch full) so torch.compile froze it.  A val/test
            # loader's final runt batch (val_size mod batch_size) has fewer
            # molecules -> the graph would emit a wrong-sized E (job 12359716:
            # pred E(64) vs batch E(48)).  PAD the runt up to the compiled n_mol
            # with isolated dummy atoms (no edges), run the fast compiled graph
            # on EVERY batch, then slice the real predictions back off.  The pad
            # atoms neither send nor receive messages, so the real atoms'
            # energies/forces are NUMERICALLY IDENTICAL to the unpadded batch.
            tgt = step._traced_nmol
            n_mol = int(batch["N"].shape[0])
            if n_mol > tgt:
                raise RuntimeError(
                    f"eval batch has n_mol={n_mol} > compiled n_mol={tgt}; set "
                    f"eval_batch_size <= batch_size so every batch fits the "
                    f"make_fx graph's baked molecule count.")
            pad_mol = tgt - n_mol
            if pad_mol:
                n_real_atom = int(batch["Z"].shape[0])
                batch = self._pad_batch_to_nmol(batch, tgt)
                if not self._logged_runt_pad:
                    logger.info("eval runt batch n_mol=%d padded to compiled "
                                "n_mol=%d (dummy atoms; predictions sliced back).",
                                n_mol, tgt)
                    self._logged_runt_pad = True
            with torch.no_grad():
                # Run with the CURRENT model's params so eval/inference sees the
                # EMA weights: _ema_apply swaps self.model to the EMA module, but a
                # REUSED training graph captured the ORIGINAL model — pass fresh
                # params explicitly (same shapes/dtypes → no recompile).  Buffers
                # are constants baked at trace time.
                E, F = step._gmc(tuple(self.model.parameters()), *step._unpack(batch))
            E, F = plainify(E), plainify(F)
            if pad_mol:
                E = E[:n_mol]           # drop padded molecules
                F = F[:n_real_atom]     # drop padded atoms
            return {"E": E, "F": F}
        return self.model(batch)

    def _pad_batch_to_nmol(self, batch, target_nmol):
        """Append isolated dummy molecules so the batch has exactly
        ``target_nmol`` molecules (= the compiled graph's baked ``n_mol``).

        Each pad molecule is one hydrogen atom with NO edges: it appears in
        ``Z``/``R``/``batch``/``N`` but is referenced by no entry of
        ``edge_index`` (left untouched, along with ``edge_shift``/``edge_mask``).
        A 0-degree atom neither sends nor receives attention messages and its
        node features evolve purely node-wise, so (a) its energy is composition-
        only and (b) it is independent of every edge ``diff`` — hence the real
        molecules' energies AND forces are bit-identical to the unpadded batch.
        Pad rows land at the END so ``pred[:n_real]`` recovers the real ones.
        """
        dev = batch["Z"].device
        n_mol = int(batch["N"].shape[0])
        pad = target_nmol - n_mol
        out = dict(batch)
        out["Z"] = torch.cat(
            [batch["Z"], torch.ones(pad, dtype=batch["Z"].dtype, device=dev)])
        out["R"] = torch.cat(
            [batch["R"],
             torch.zeros(pad, batch["R"].shape[1], dtype=batch["R"].dtype, device=dev)])
        out["batch"] = torch.cat(
            [batch["batch"],
             torch.arange(n_mol, target_nmol, dtype=batch["batch"].dtype, device=dev)])
        out["N"] = torch.cat(
            [batch["N"], torch.ones(pad, dtype=batch["N"].dtype, device=dev)])
        # edge_index / edge_shift / edge_mask unchanged — pad atoms have no edges.
        return out

    def _evaluate(self, loader):
        """Evaluate on a loader. Returns (loss, energy_mae, force_mean_l2, force_mae).

        The three metrics are pooled over every molecule / atom of the loader
        (see :meth:`_compute_metrics`).  ``loss`` is the mean of the per-batch
        training losses -- unchanged, because it drives the LR scheduler and
        the best-checkpoint selection of every reported run.
        """
        self.model.eval()
        # Defragment the allocator pool before eval's larger (eval_batch) forwards
        # so reserved-but-unallocated holes from the train step don't trigger a
        # mid-eval OOM over a long run.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tot_loss, tot_e_abs, tot_f_l2, tot_f_abs = 0.0, 0.0, 0.0, 0.0
        tot_mol, tot_atom, n_batch = 0, 0, 0
        # no_grad so the loss/metric ops build no throwaway graph.  model.forward
        # re-enables grad internally for the force autograd.grad and detaches its
        # outputs in eval mode, so the forces still come out correct.
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                pred = self._forward(batch)
                loss, _, _ = self.combined_loss(pred, batch, self.rho_force)
                e_abs, f_l2, f_abs, n_mol, n_atom = self._compute_metrics(pred, batch)
                tot_loss += loss.item()
                tot_e_abs += e_abs
                tot_f_l2 += f_l2
                tot_f_abs += f_abs
                tot_mol += n_mol
                tot_atom += n_atom
                n_batch += 1
        self.model.train()
        # Release any eval-only allocations / fragmentation so they don't
        # accumulate alongside the compiled training step across a long run.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (tot_loss / max(n_batch, 1),
                tot_e_abs / max(tot_mol, 1),
                tot_f_l2 / max(tot_atom, 1),
                tot_f_abs / max(3 * tot_atom, 1))

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path):
        torch.save({
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "warmup_scheduler": (self._warmup_scheduler.state_dict()
                                 if self._warmup_scheduler is not None else None),
            "best_val_loss": self.best_val_loss,
            "no_improve_count": self._no_improve_count,
            "ema_model": (self._ema_model.state_dict()
                          if self._ema_model is not None else None),
        }, path)

    def _check_same_optimizer(self, saved, path):
        """Refuse a checkpoint whose optimizer state comes from another optimizer.

        Such state cannot be loaded (different param-group layout), and the
        fallback in load_checkpoint would then carry on silently with fresh
        AdamW moments, i.e. as a different training run.  A checkpoint written
        by AdamW before the block-parameter refactor has the same single group
        and AdamW's settings, so it still takes that fallback.
        """
        if not isinstance(saved, dict):
            return
        groups = saved.get("param_groups") or []
        known = set(self.optimizer.param_groups[0]) | {"params"}
        foreign = sorted({k for g in groups for k in g} - known)
        if len(groups) != len(self.optimizer.param_groups) and foreign:
            raise ValueError(
                f"{path}: its optimizer state was written by a different optimizer "
                f"({len(groups)} parameter groups, with settings AdamW does not have: "
                f"{', '.join(foreign)}).  This code trains with AdamW only, so the run "
                f"cannot be resumed; its weights can still be scored with evaluate.py.")

    def _checkpoint_lr(self, ckpt):
        """Learning rate a checkpoint was at, for when its optimizer state cannot
        be loaded: its saved param group's lr; failing that, the warmup value
        at its step (the configured lr once warmup is over)."""
        groups = (ckpt.get("optimizer") or {}).get("param_groups") or []
        if groups and isinstance(groups[0].get("lr"), (int, float)):
            return float(groups[0]["lr"])
        base = self.optimizer.param_groups[0].get("initial_lr",
                                                  self.optimizer.param_groups[0]["lr"])
        step = ckpt.get("step", 0)
        if self._warmup_scheduler is None or step >= self.warmup_steps:
            return base
        start = self._warmup_scheduler.start_factor
        return base * (start + (1.0 - start) * step / self.warmup_steps)

    def load_checkpoint(self, path):
        # weights_only=False: full training checkpoints also hold optimizer /
        # scheduler state (trusted files written by save_checkpoint); explicit
        # so torch >= 2.6 (default weights_only=True) behaves like 2.5.
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._check_same_optimizer(ckpt.get("optimizer"), path)
        self.model.load_state_dict(ckpt["model"])
        # The equivariant linears now store ONE PARAMETER PER BLOCK, so a
        # checkpoint written before that refactor has a different parameter
        # COUNT.  torch's optimizer state is positional, so it cannot be mapped
        # onto the new parameter list.  Model weights still load exactly (a
        # load_state_dict pre-hook splits the old flat tensors block-wise); only
        # the optimizer moments are lost, which costs a short re-adaptation
        # rather than the run.  Failing hard here would strand every existing
        # checkpoint.
        try:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError, RuntimeError) as exc:
            # The learning rate lives in the optimizer's param groups, so it was
            # not restored either: it still holds the warmup START value
            # (lr * 1e-8), and past warmup nothing would raise it again (the
            # plateau scheduler only lowers it).  Take it from the checkpoint.
            lr = self._checkpoint_lr(ckpt)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
            logger.warning(
                "Could not restore optimizer state (%s: %s). Model weights WERE "
                "restored exactly; optimizer moments restart from zero; learning "
                "rate set to %.3g. Expected when resuming a checkpoint written "
                "before the block-parameter refactor.", type(exc).__name__, exc, lr)
        if ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("warmup_scheduler") is not None and self._warmup_scheduler is not None:
            self._warmup_scheduler.load_state_dict(ckpt["warmup_scheduler"])
        if ckpt.get("ema_model") is not None and self._ema_model is not None:
            if self._in_swa_stage(ckpt.get("step", 0)):
                # Written inside the SWA stage: its averaged model is the UNIFORM
                # average started by _enter_swa_stage, not the EMA built in
                # __init__.  Rebuild it as uniform so later update_parameters()
                # calls keep averaging uniformly (the state_dict keys, including
                # n_averaged, are the same for both kinds).
                self._ema_model = torch.optim.swa_utils.AveragedModel(self.model)
                logger.info("Checkpoint is inside the SWA stage (step %d > swa_start "
                            "%d): restoring a uniform SWA average",
                            ckpt.get("step", 0), self.swa_start)
            self._ema_model.load_state_dict(ckpt["ema_model"])
        self.step = ckpt.get("step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self._no_improve_count = ckpt.get("no_improve_count", 0)
        logger.info("Resumed from step %d", self.step)

    # ------------------------------------------------------------------
    # wandb helper
    # ------------------------------------------------------------------

    def _log_wandb(self, metrics):
        if self._wandb is not None:
            self._wandb.log(metrics, step=self.step)

    # ------------------------------------------------------------------
    # Fully-dynamic make_fx + torch.compile training step (opt-in)
    # ------------------------------------------------------------------

    def _makefx_backward_step(self, batch):
        """One training step via the fully-dynamic make_fx + torch.compile graph.

        Builds the compiled graph lazily on first use (warm Wigner cache → symbolic
        make_fx → torch.compile(dynamic=True)); ONE graph serves every batch shape.
        The compiled E/F may carry a symbolic tag — safe for loss.backward()/
        optimizer.step() (pure tensor ops), but we plainify loss+pred afterwards so
        the downstream .item()/metrics don't trip GuardOnDataDependentSymNode.
        """
        from .backward.makefx_compiled import (
            MakeFXCompiledStep, MakeFXForwardModule, plainify)
        if self._makefx_step is None:
            self._makefx_step = MakeFXCompiledStep(self.model, batch)
            logger.info("make_fx step compiled: %d graph nodes, %d complex ops, "
                        "%d detach/alias stripped",
                        self._makefx_step._n_nodes, self._makefx_step._n_complex,
                        self._makefx_step._n_detach)
            if self.is_dist:
                from torch.nn.parallel import DistributedDataParallel as DDP
                # Wrap the compiled forward in an nn.Module so DDP.forward() arms
                # the gradient reducer; the single loss.backward() below then
                # drives DDP's bucketed all-reduce (validated: ddp_smoke_dev.py).
                self._ddp_module = DDP(
                    MakeFXForwardModule(self._makefx_step),
                    device_ids=[self.gpu_index],
                    gradient_as_bucket_view=True,
                )
                logger.info("DDP-wrapped make_fx forward (device_ids=[%d], "
                            "gradient_as_bucket_view=True)", self.gpu_index)
        if self.is_dist:
            E, F = self._ddp_module(batch)            # DDP.forward -> arms reducer
            pred = {"E": E, "F": F}
        else:
            pred = self._makefx_step.step(batch)      # tagged, grad-connected
        loss, _, _ = self.combined_loss(pred, batch, self.rho_force)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        # plainify so downstream logging (.item()) / metrics don't trip the tag
        loss = plainify(loss)
        pred = {k: plainify(v) for k, v in pred.items()}
        return pred, loss

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(self, max_iterations, train_loader, val_loader=None, test_loader=None):
        """Train for ``max_iterations`` gradient steps.

        Parameters
        ----------
        max_iterations : int
            Total number of gradient steps to run.
        train_loader : DataLoader
        val_loader : DataLoader | None
        test_loader : DataLoader | None
        """
        target_step = max_iterations
        train_iter = iter(train_loader)
        self.model.train()

        running_loss_sum = torch.tensor(0.0, device=self.device)
        running_n = 0
        t0 = time.time()

        from torch.utils.data.distributed import DistributedSampler
        # A resumed run starts the bar at the resumed step (one update per step below).
        pbar = (tqdm(total=max_iterations, initial=self.step, desc="Training", unit="step",
                     miniters=100, mininterval=0, maxinterval=float("inf"))
                if self.is_main else None)
        epoch = 0
        # Resumed into the SWA/energy stage: restore the energy-heavy loss weighting
        # (LR / scheduler / uniform averaged model already reflect stage 2 via the
        # checkpoint, see load_checkpoint).  A checkpoint at exactly step ==
        # swa_start predates the switch, so the loop below performs it.
        if self._in_swa_stage(self.step):
            self.rho_force = self.swa_rho_force
            self._swa_entered = True
        while self.step < target_step:
            # MACE-style two-stage SWA: fire the stage-2 switch once at swa_start.
            if self.swa_start > 0 and not self._swa_entered and self.step >= self.swa_start:
                self._enter_swa_stage()
            # Fetch next batch, cycle the loader when exhausted
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                _sampler = getattr(train_loader, "sampler", None)
                if self.is_dist and isinstance(_sampler, DistributedSampler):
                    _sampler.set_epoch(epoch)   # reshuffle differently each epoch
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            # ── training step dispatch ─────────────────────────────────
            if self.use_makefx:
                # make_fx + torch.compile: forward + force 1st-backward baked into
                # ONE graph, so training is a single loss.backward() (dynamic shapes,
                # DDP-compatible).  The production path.
                pred, loss = self._makefx_backward_step(batch)
            else:
                # Plain eager-autograd fallback: model.forward computes the force via
                # autograd.grad(E, R, create_graph=True); loss.backward() then
                # gives the param grads (the standard force-matching 2nd backward).
                pred = self.model(batch)
                loss, _, _ = self.combined_loss(pred, batch, self.rho_force)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            self._ema_update()

            if self._warmup_scheduler is not None and self.step < self.warmup_steps:
                self._warmup_scheduler.step()

            self.step += 1
            if pbar is not None:
                pbar.update(1)
            # Accumulate loss on-device — no .item() sync per step
            running_loss_sum += loss.detach()
            running_n += 1

            # ---- Eval --------------------------------------------------
            if self.step % self.eval_every == 0 or self.step == target_step:
                # Single CUDA sync point: pull running loss to CPU (rank-local)
                dt = time.time() - t0
                avg_loss = running_loss_sum.item() / max(running_n, 1)
                # Train metrics from the last batch use the TRAINING weights, so
                # compute them before the EMA swap (rank 0 logs them).
                if self.is_main:
                    e_abs, f_l2, f_abs, n_mol, n_atom = self._compute_metrics(pred, batch)
                    train_energy_mae = e_abs / max(n_mol, 1)
                    train_force_l2 = f_l2 / max(n_atom, 1)
                    train_force_mae = f_abs / max(3 * n_atom, 1)
                self._ema_apply()

                # Rank 0 alone evaluates; the plateau-scheduler metric is
                # broadcast so EVERY rank steps the scheduler with the identical
                # value — otherwise their LR (and Adam moments) would diverge.
                val_metrics = None
                if val_loader is not None:
                    if self.is_main:
                        val_metrics = self._evaluate(val_loader)
                    sched_metric = self._bcast_scalar(
                        val_metrics[0] if val_metrics is not None else 0.0)
                else:
                    sched_metric = self._bcast_scalar(avg_loss)

                if self.step >= self.warmup_steps:
                    self.scheduler.step(sched_metric)

                # Logging / best-save / test eval happen on rank 0 only.
                if self.is_main:
                    msg = (f"Step {self.step:7d}/{target_step} | "
                           f"loss {avg_loss:.4e}  E_mae {train_energy_mae:.4e}  "
                           f"F_l2 {train_force_l2:.4e}  F_mae {train_force_mae:.4e} | "
                           f"lr {self.optimizer.param_groups[0]['lr']:.2e} | {dt:.1f}s")
                    wb = {
                        "train/loss": avg_loss,
                        "train/energy_mae": train_energy_mae,
                        "train/force_l2": train_force_l2,
                        "train/force_mae": train_force_mae,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                    if val_metrics is not None:
                        val_loss, val_energy_mae, val_force_l2, val_force_mae = val_metrics
                        msg += (f" | val {val_loss:.4e}  E_mae {val_energy_mae:.4e}  "
                                f"F_l2 {val_force_l2:.4e}  F_mae {val_force_mae:.4e}")
                        wb["val/loss"] = val_loss
                        wb["val/energy_mae"] = val_energy_mae
                        wb["val/force_l2"] = val_force_l2
                        wb["val/force_mae"] = val_force_mae
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self._no_improve_count = 0
                            best_name = f"{self._wandb_name}_best.pt" if self._wandb_name else "best.pt"
                            # Save EMA weights as the best model
                            torch.save(self.model.state_dict(),
                                       os.path.join(self.out_dir, best_name))
                        else:
                            self._no_improve_count += 1
                    if test_loader is not None:
                        test_loss, test_energy_mae, test_force_l2, test_force_mae = self._evaluate(test_loader)
                        msg += (f" | test {test_loss:.4e}  E_mae {test_energy_mae:.4e}  "
                                f"F_l2 {test_force_l2:.4e}  F_mae {test_force_mae:.4e}")
                        wb["test/loss"] = test_loss
                        wb["test/energy_mae"] = test_energy_mae
                        wb["test/force_l2"] = test_force_l2
                        wb["test/force_mae"] = test_force_mae
                    self._log_wandb(wb)
                    logger.info(msg)

                self._ema_restore()

                running_loss_sum.zero_()
                running_n = 0
                t0 = time.time()

            # ---- Checkpoint (rank 0 only) ------------------------------
            if self.step % self.save_every == 0 and self.is_main:
                ckpt_name = f"{self._wandb_name}.pt" if self._wandb_name else "checkpoint.pt"
                self.save_checkpoint(
                    os.path.join(self.out_dir, ckpt_name),
                )

        if pbar is not None:
            pbar.close()
        if self.is_main:
            final_name = f"{self._wandb_name}_final.pt" if self._wandb_name else "final.pt"
            self.save_checkpoint(os.path.join(self.out_dir, final_name))
            logger.info("Training complete. Checkpoints saved to %s", self.out_dir)

            if self._wandb is not None:
                self._wandb.finish()

        if self.is_dist:
            import torch.distributed as dist
            dist.barrier()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def inference(self, test_loader):
        """Evaluate the model on a test set and log results.

        Uses EMA weights when available.

        Returns
        -------
        tuple of (loss, energy_mae, force_mean_l2, force_mae)
        """
        self._ema_apply()
        test_loss, test_energy_mae, test_force_l2, test_force_mae = self._evaluate(test_loader)
        self._ema_restore()
        logger.info("Inference | loss %s  E_mae %s  F_l2 %s  F_mae %s",
                     f"{test_loss:.4e}", f"{test_energy_mae:.4e}",
                     f"{test_force_l2:.4e}", f"{test_force_mae:.4e}")
        return test_loss, test_energy_mae, test_force_l2, test_force_mae

    def predict(self, loader):
        """Run inference and return total energies and forces.

        Adds atomic reference energies back to the predicted interaction
        energies so the output is in absolute (total) energy units.

        Returns
        -------
        dict with keys ``E`` (list of per-molecule total energies) and
        ``F`` (list of per-atom force tensors).
        """
        self._ema_apply()
        self.model.eval()
        all_E, all_F = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                pred = self._forward(batch)
                E_total = self.model.add_atomic_reference(
                    pred["E"], batch["Z"], batch["batch"], batch["N"].shape[0],
                )
                all_E.append(E_total.detach().cpu())
                all_F.append(pred["F"].detach().cpu())
        self.model.train()
        self._ema_restore()
        return {"E": torch.cat(all_E), "F": torch.cat(all_F)}

    # ------------------------------------------------------------------
    # Factory: build everything from a flat config dict
    # ------------------------------------------------------------------

    @staticmethod
    def build_model(config):
        """Build the :class:`AtomTrick` model described by a flat config dict.

        Returns the model on CPU in float32 (callers move / cast it).  Keys
        match the ``train.py`` CLI (``args.json`` files).  Used by
        :meth:`from_config` and by ``train.py --dry_run``.

        Missing keys fall back to the behaviour the code had *before* that
        option existed -- NOT to the paper configuration -- so that an old
        ``args.json`` (which lacks the newer keys) rebuilds exactly the model
        it was trained with.  To get the paper model, pass a complete config
        (``train.py`` always writes one) or use ``train.py``'s defaults.
        ``radial_rank`` / ``energy_correlation`` are dead options and ignored,
        and so are the keys of options this code version does not have (an
        older ``args.json`` may carry them).  A missing
        ``escn_frame`` means ``"polefree"``.  Raises ``ValueError`` if
        ``edge_cutoff`` and ``cutoff`` differ (:func:`check_cutoffs`).
        """
        check_cutoffs(config)
        return AtomTrick(
            irreps_hidden=config.get("irreps", "16x0e + 16x1o + 16x2e"),
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 4),
            ff_mult=config.get("ff_mult", 4),
            max_z=config.get("max_z", 100),
            num_radial_basis=config.get("num_radial_basis", 16),
            cutoff=config.get("cutoff", 5.0),
            radial_hidden=config.get("radial_hidden", 64),
            energy_hidden=config.get("energy_hidden", 64),
            radial_mode=config.get("radial_mode", "gate"),
            k_mode=config.get("k_mode") or config.get("kv_mode") or "o2",
            v_mode=config.get("v_mode") or config.get("kv_mode") or "o2",
            ffn_act_mode=config.get("ffn_act_mode", "gate"),
            rbf_type=config.get("rbf_type", "expnorm"),
            layernorm_mode=config.get("layernorm_mode", "default"),
            attn_nonlin=config.get("attn_nonlin", "softmax"),
            atomic_ref=config.get("atomic_ref"),
            escn_frame=config.get("escn_frame", "polefree"),
        )

    @classmethod
    def from_config(cls, config):
        """Construct a Trainer from a flat config dict (e.g. argparse namespace).

        Expected keys match the CLI arguments in train.py.  Missing model keys
        use legacy fallbacks (see :meth:`build_model`), so old ``args.json``
        files rebuild their model.  ``matmul_precision`` defaults to ``"highest"``
        (full fp32 matmuls; ``evaluate.py`` passes ``"high"`` explicitly for a
        config that records no precision).  Unless ``config["_skip_save_args"]`` is
        set, rank 0 writes the config plus a ``_provenance`` block to
        ``<out_dir>/args.json``.
        """
        seed = config.get("seed", 42)
        torch.manual_seed(seed)

        # ---- Distributed setup (torchrun/srun sets WORLD_SIZE/RANK/LOCAL_RANK) --
        # Init the process group and pin this rank to its GPU BEFORE the model is
        # built/moved, so every param lands on cuda:LOCAL_RANK.  Same seed on all
        # ranks => identical init (DDP also broadcasts rank-0 params at wrap).
        _world_size = int(os.environ.get("WORLD_SIZE", "1"))
        _rank = int(os.environ.get("RANK", "0"))
        _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if _world_size > 1:
            import torch.distributed as dist
            # nccl for real multi-GPU; ATOMTRICK_DDP_BACKEND=gloo lets a dev-box
            # smoke run 2 ranks on ONE GPU (nccl needs 1 rank/GPU).
            backend = os.environ.get("ATOMTRICK_DDP_BACKEND", "nccl")
            if not dist.is_initialized():
                # Harden against transient shared-filesystem I/O stalls: one rank stalling
                # on a checkpoint-save/eval I/O spike must not trip NCCL's 600s
                # collective-timeout watchdog and kill a multi-day run (as happened
                # to a multi-GPU OC20 run at step ~402k). Overridable
                # via ATOMTRICK_DDP_TIMEOUT_MIN (default 30 min).
                _ddp_timeout_min = int(os.environ.get("ATOMTRICK_DDP_TIMEOUT_MIN", "30"))
                dist.init_process_group(
                    backend=backend,
                    timeout=timedelta(minutes=_ddp_timeout_min))
            _ndev = max(1, torch.cuda.device_count())
            _dev_idx = _local_rank % _ndev   # all ranks -> cuda:0 when 1 GPU visible
            torch.cuda.set_device(_dev_idx)
            config = dict(config)
            config["device"] = f"cuda:{_dev_idx}"
            logger.info("DDP from_config: world_size=%d rank=%d backend=%s -> device cuda:%d",
                        _world_size, _rank, backend, _dev_idx)

        device_str = config.get("device")
        if device_str is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_str)
        logger.info("Device: %s", device)

        out_dir = config.get("out_dir", "checkpoints")
        os.makedirs(out_dir, exist_ok=True)
        if not config.get("_skip_save_args", False) and _rank == 0:
            saved = dict(config)
            # Provenance: code version + software/hardware (no hostname).
            # Values supplied by the caller (train.py adds the resolved
            # make_fx switches) take precedence.
            saved["_provenance"] = {**collect_provenance(device),
                                    **(config.get("_provenance") or {})}
            with open(os.path.join(out_dir, "args.json"), "w") as f:
                json.dump(saved, f, indent=2)

        # ---- Precision -------------------------------------------------------
        dtype_str = config.get("dtype", "float32")
        dtype = torch.float64 if dtype_str == "float64" else torch.float32

        # ---- Model -----------------------------------------------------------
        model = cls.build_model(config).to(device=device, dtype=dtype)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Model parameters: %s  dtype: %s", f"{n_params:,}", dtype)

        if dtype == torch.float32:
            # "highest" = full fp32 matmuls, the default.  "high" = TF32 matmuls on
            # Ampere+ GPUs (a config that records no precision is resolved by evaluate.py).
            precision = config.get("matmul_precision") or "highest"
            torch.set_float32_matmul_precision(precision)
            logger.info("float32 matmul precision: %s", precision)

        return cls(
            model,
            device=device,
            out_dir=out_dir,
            rho_force=config.get("rho_force", 0.5),
            grad_clip=config.get("grad_clip", 10.0),
            save_every=config.get("save_every", 1000),
            eval_every=config.get("eval_every", 500),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-5),
            optimizer=config.get("optimizer", "adamw"),
            scheduler_patience=config.get("scheduler_patience", 10),
            scheduler_factor=config.get("scheduler_factor", 0.5),
            scheduler_min_lr=config.get("scheduler_min_lr", 1e-6),
            warmup_steps=config.get("warmup_steps", 1000),
            wandb_project=config.get("wandb_project"),
            wandb_name=config.get("wandb_name"),
            ema_decay=config.get("ema_decay", 0.0),
            swa_start=config.get("swa_start", 0),
            swa_rho_force=config.get("swa_rho_force", 0.1),
            swa_lr=config.get("swa_lr", 0.0),
        )

    @staticmethod
    def build_loaders(config, max_edges=None):
        """Build train/val/test data loaders from a config dict.

        Parameters
        ----------
        config : dict
        max_edges : int, optional
            Pad edge_index to this size (static-shape batches; the model
            masks the padding via ``edge_mask``).

        Returns
        -------
        tuple of (train_loader, val_loader or None, test_loader or None)
        """
        from torch.utils.data import Subset

        edge_cutoff = config.get("edge_cutoff", 5.0)
        max_neighbors = config.get("max_neighbors")
        dtype_str = config.get("dtype", "float32")
        dtype = torch.float64 if dtype_str == "float64" else torch.float32
        atomic_ref = config.get("atomic_ref")
        ds_kwargs = dict(edge_cutoff=edge_cutoff, dtype=dtype,
                         atomic_ref=atomic_ref, max_neighbors=max_neighbors)
        train_ds = MLFFDataset(config["data"], **ds_kwargs)
        val_ds = None
        test_ds = None

        if config.get("val_data") is not None:
            val_ds = MLFFDataset(config["val_data"], **ds_kwargs)
        elif config.get("val_split", 0) > 0:
            n_total = len(train_ds)
            n_val = int(n_total * config["val_split"])
            indices = torch.randperm(n_total).tolist()
            val_ds = Subset(train_ds, indices[:n_val])
            train_ds = Subset(train_ds, indices[n_val:])

        if config.get("test_data") is not None:
            test_ds = MLFFDataset(config["test_data"], **ds_kwargs)

        train_batch_size = config.get("batch_size", 32)
        eval_batch_size = config.get("eval_batch_size") or train_batch_size
        loader_cfg = {
            "batch_size": train_batch_size,
            "num_workers": config.get("num_workers", 0),
        }
        eval_loader_cfg = {
            "batch_size": eval_batch_size,
            "num_workers": config.get("num_workers", 0),
        }
        # Under DDP each rank draws a disjoint shard (DistributedSampler); the
        # per-rank batch_size stays as configured so effective batch =
        # world_size * batch_size.  Val/test are NOT sharded (rank 0 evaluates
        # the full set).
        _world_size = int(os.environ.get("WORLD_SIZE", "1"))
        _rank = int(os.environ.get("RANK", "0"))
        train_loader = build_dataloader(train_ds, loader_cfg, train=True,
                                         max_edges=max_edges,
                                         distributed=_world_size > 1,
                                         rank=_rank, world_size=_world_size)
        val_loader = (build_dataloader(val_ds, eval_loader_cfg, train=False)
                      if val_ds is not None else None)
        test_loader = (build_dataloader(test_ds, eval_loader_cfg, train=False)
                       if test_ds is not None else None)

        logger.info("Train samples: %d", len(train_ds))
        if val_ds is not None:
            logger.info("Val   samples: %d", len(val_ds))
        if test_ds is not None:
            logger.info("Test  samples: %d", len(test_ds))

        return train_loader, val_loader, test_loader
