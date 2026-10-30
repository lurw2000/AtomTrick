"""Fully-dynamic compiled training step via ``make_fx`` + ``torch.compile``.

Rationale
---------
torch.compile cannot compile *across* a ``create_graph=True`` double-backward
(pytorch#91469).  ``make_fx`` traces the forward **and** the force 1st-backward
(``autograd.grad(E, R, create_graph=True)``) into ONE flat aten graph, so training
becomes a *single* ``loss.backward()`` that torch.compile CAN compile.  Tracing in
``symbolic`` mode yields ONE graph that serves every OC20 batch shape — no per-shape
recompile storm (validated: ``unique_graphs==1`` across shapes).

CRITICAL — warm the Wigner cache before tracing
-----------------------------------------------
The module-level Wigner cache (``o3.wigner._wigner_D_module_cache``) is cold on the
first forward.  If ``make_fx`` traces that first call it bakes the one-time
construction of the *complex* change-of-basis matrix ``U`` (``_change_of_basis_
real_complex``) into the graph — ~111 complex aten ops.  Inductor cannot codegen
complex ops → eager fallback → under DYNAMIC shapes it MIS-compiles → NaN forces.
Running one eager forward first populates the cache so the traced forward reads only
the real ``_U_re/_U_im`` buffers (``complex_nodes==0``).

Requirements (set in the launch env, BEFORE importing AtomTrick)
-----------------------------------------------------------------
``PYTORCH_JIT=0`` — TorchScript breaks make_fx's 2nd-order trace.

Numerics vs eager (fp32): forward E ~1e-7, forces F ~1e-6 (bit-exact); param grads
relL2 ~1e-5, cosine 1.0 (training-identical).

CRITICAL — strip the traced ``aten.detach`` nodes
-------------------------------------------------
Tracing a backward under ``__torch_dispatch__`` records the autograd engine's
internal ``aten.detach`` calls into the graph, where they become hard gradient
barriers for the *second* backward.  Forces stay bit-exact (detach is an identity
on data) while ``d(force-loss)/dtheta`` silently loses terms — 4.4e-2 relative
error, fp64, on the L2 test model.  :func:`_strip_detach` removes them; see its
docstring for the measurement and :meth:`MakeFXCompiledStep._check_grad_connected`
for the guard.  Fixed 2026-07-29; runs traced before that date had this error.

Symbolic-tag boundary
---------------------
Compiled outputs may carry an unbacked-symfloat *tag*; tensor math and
``loss.backward()``/``optimizer.step()`` are unaffected, but extracting a python
scalar (``.item()``/``.numpy()``) trips ``GuardOnDataDependentSymNode``.  Use
:func:`plainify` at the logging / metric boundary.
"""

import os

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.func import functional_call


def _maybe_enable_makefx_duck_off():
    """Disable make_fx duck-shaping (env-gated) for single-molecule datasets.

    rMD17-style data (every batch = same molecule) has a CONSTANT node/edge count
    that can EQUAL a model constant (channel mult=64, or total irrep dim=1024). In
    the default symbolic tracer, ``make_fx`` also symbolizes the model *parameters*
    and DUCK-SHAPES equal-valued dims onto one symbol — so a param dim of 64 shares a
    symbol with the node count of 64, gets forced concrete, and bakes a fixed size
    (``view [64,1024]``) that crashes on a different-count batch.  Setting
    ``duck_shape=False`` on the (still symbolic) ShapeEnv keeps those dims distinct
    (node reshape stays ``[sym,1024]``) while SymInts still thread through every op —
    unlike the ``tracing_mode="real"`` fake plumbing, which bakes ~the whole graph.

    Env-gated via ``ATOMTRICK_MAKEFX_DUCKOFF=1`` so OC20/SPICE (large VARIED counts,
    never collide) keep the default.  Validated (L40S, bs4, L6): fully dynamic for the
    colliding molecules (salicylic/naphthalene), ~3.5x faster than eager,
    ``unique_graphs==2`` (no recompile storm), and NO speed regression vs duck-on on
    non-colliding molecules.
    """
    if os.environ.get("ATOMTRICK_MAKEFX_DUCKOFF", "0") != "1":
        return
    import torch.fx.experimental.symbolic_shapes as _ss
    if getattr(_ss.ShapeEnv, "_atomtrick_duckoff", False):
        return                              # already patched (idempotent)
    _orig_init = _ss.ShapeEnv.__init__
    def _init(self, *a, **k):
        k["duck_shape"] = False
        return _orig_init(self, *a, **k)
    _ss.ShapeEnv.__init__ = _init
    _ss.ShapeEnv._atomtrick_duckoff = True


def _strip_detach(gm):
    """Remove the ``aten.detach`` / ``aten.alias`` nodes make_fx records while
    tracing the force ``autograd.grad(E, R, create_graph=True)``.

    **This is a correctness fix, not an optimisation.**  Running a backward under
    ``__torch_dispatch__`` makes the autograd engine's internal saved-variable
    bookkeeping visible to the tracer, so ``aten.detach`` calls land in the graph
    (66 of them for the L2 model).  In eager they are engine-internal and do NOT
    sever the double backward — with ``create_graph=True`` the engine reattaches
    ``grad_fn``.  Baked into an FX graph and *replayed*, each one becomes a hard
    gradient barrier: the forward values and the forces are unaffected (detach is
    an identity on data), but the SECOND backward loses every term that flows
    through them.

    Measured on the L2 test model in fp64: with the detaches left in,
    ``d(force-loss)/dtheta`` was wrong by 4.4e-2 relative (finite differences side
    with eager: eager 9.5e-8, traced 4.3e-2); after stripping, 3.2e-16.  Forces
    are bit-identical either way, which is exactly why this hid for so long.
    Independent of tracing_mode (``symbolic``/``real``/``fake`` all showed it) and
    of inductor (the raw GraphModule and the compiled one agreed to 4 digits).

    Replacing a detach with its input is value-identical, so the only hazard is
    an in-place op writing through the detached view; the graph is checked by
    :meth:`MakeFXCompiledStep._check_grad_connected` right afterwards.
    """
    targets = (torch.ops.aten.detach.default, torch.ops.aten.alias.default)
    removed = 0
    for node in reversed(list(gm.graph.nodes)):
        if node.op == "call_function" and node.target in targets:
            node.replace_all_uses_with(node.args[0])
            gm.graph.erase_node(node)
            removed += 1
    if removed:
        gm.graph.lint()
        gm.recompile()
    return removed


def plainify(t):
    """Copy a (possibly symbolic-tagged) compiled output into a genuinely plain
    tensor so ``.item()``/metrics work.  ``copy_`` reads the real runtime data."""
    p = torch.empty(t.shape, dtype=t.dtype, device=t.device)
    p.copy_(t.detach())
    return p


class MakeFXCompiledStep:
    """Builds (lazily, from a sample batch) and caches ONE compiled dynamic graph
    for forward + forces, reused across all batch shapes.

    Call ``step(batch)`` for a grad-connected ``{"E","F"}`` (safe for
    ``loss.backward()``); plainify before reading python scalars.
    """

    def __init__(self, model, sample_batch, dynamic=True, train_mode=True):
        self.model = model
        self._train_mode = train_mode
        self._pnames = [n for n, _ in model.named_parameters()]
        self._pbufs = dict(model.named_buffers())
        self._pvals = tuple(model.parameters())
        self._has_shift = sample_batch.get("edge_shift", None) is not None
        # Molecule count baked into the graph.  make_fx makes atom/edge counts
        # dynamic (they vary every step), but n_mol = batch["N"].shape[0] is
        # CONSTANT during training (= batch_size; train uses drop_last=True) so
        # torch.compile specialises the per-molecule energy output torch.zeros(
        # n_mol) to this value.  A batch with a different molecule count (e.g. an
        # eval loader's final runt batch, val_size mod batch_size) would get a
        # wrong-sized E — callers compare against _traced_nmol and PAD the runt
        # up to it with isolated dummy atoms, then slice the real predictions
        # back off.  (Root cause of the 12359716 first-val crash: pred E(64) vs
        # batch E(48).)
        self._traced_nmol = int(sample_batch["N"].shape[0])
        was_training = model.training
        # train_mode=True bakes create_graph=True (the force 2nd-order graph the
        # training loss.backward() needs).  train_mode=False traces the EVAL forward
        # (create_graph=False, detached E/F) -> a leaner, faster INFERENCE graph.
        model.train(train_mode)
        args = self._unpack(sample_batch)
        # A plain closure (NOT a bound method): make_fx introspects co_argcount and
        # a bound method's `self` would make it expect one extra argument.
        fwd = self._make_fwd()
        # warm the Wigner module cache -> traced forward reads only real buffers
        _ = fwd(self._pvals, *args)
        # single-molecule (rMD17) data counts can duck-collide with model constants
        # and bake fixed shapes -> crash; ATOMTRICK_MAKEFX_DUCKOFF=1 disables duck-shaping.
        _maybe_enable_makefx_duck_off()
        gm = make_fx(fwd, tracing_mode="symbolic",
                     _allow_non_fake_inputs=True)(self._pvals, *args)
        self._n_complex = _count_complex(gm)
        if self._n_complex:
            raise RuntimeError(
                f"make_fx graph has {self._n_complex} complex ops — the Wigner cache "
                "was not warm before tracing (see makefx_compiled docstring).")
        self._n_detach = _strip_detach(gm)
        self._check_grad_connected(gm, args)
        self._gmc = torch.compile(gm, dynamic=dynamic)
        self.gm = gm                # raw aten GraphModule (pre-compile, for inspection)
        self._n_nodes = sum(1 for _ in gm.graph.nodes)
        model.train(was_training)   # restore caller's mode

    def _check_grad_connected(self, gm, args):
        """Guard the detach strip: every parameter must receive a force gradient.

        A detach left in the graph silently *drops* second-order terms — the
        forces still come out right, so nothing downstream notices.  This is the
        cheap invariant that would have caught it: differentiate a dummy force
        loss and require that every parameter with ``requires_grad`` gets a
        finite, non-None gradient.  (It cannot detect a *partial* loss, hence
        the strip is unconditional and this is a backstop, not the test.)
        """
        _E, F = gm(self._pvals, *args)
        missing = torch.autograd.grad(
            F.pow(2).sum(), self._pvals, allow_unused=True, retain_graph=False)
        bad = [n for n, (p, g) in zip(self._pnames, zip(self._pvals, missing))
               if p.requires_grad and (g is None or not torch.isfinite(g).all())]
        if bad:
            raise RuntimeError(
                "make_fx graph is not fully grad-connected to the parameters "
                f"({len(bad)} without a finite force gradient, e.g. {bad[:3]}). "
                "The traced force backward is incomplete — do not train on it.")

    def _unpack(self, batch):
        esh = batch.get("edge_shift", None)
        if (esh is not None) != self._has_shift:
            raise RuntimeError(
                "edge_shift presence changed between trace and call — the make_fx "
                "graph is shape/structure specialised; keep the dataset homogeneous.")
        return (batch["Z"], batch["R"], batch["edge_index"], batch["batch"],
                batch["N"], esh)

    def _make_fwd(self):
        """Return a PLAIN closure (not a bound method) so make_fx's arg-count
        introspection matches the concrete args passed."""
        pnames, pbufs, model = self._pnames, self._pbufs, self.model

        def fwd(pvals, Z, R, ei, mol, N, esh):
            params = dict(zip(pnames, pvals))
            params.update(pbufs)
            b = {"Z": Z, "R": R, "edge_index": ei, "batch": mol, "N": N}
            if esh is not None:
                b["edge_shift"] = esh
            out = functional_call(model, params, (b,))
            return out["E"], out["F"]

        return fwd

    def step(self, batch):
        """Grad-connected compiled forward. Returns ``{"E","F"}`` (may carry the
        symbolic tag — safe for ``loss.backward()``; plainify before ``.item()``)."""
        E, F = self._gmc(self._pvals, *self._unpack(batch))
        return {"E": E, "F": F}


class MakeFXForwardModule(torch.nn.Module):
    """nn.Module wrapper so the make_fx compiled forward can be driven through
    ``torch.nn.parallel.DistributedDataParallel``.

    DDP arms its gradient-sync hooks inside ``DDP.forward()``; calling the compiled
    callable directly (:meth:`MakeFXCompiledStep.step`) bypasses that, so the
    reducer never fires.  Registering ``model`` as a submodule makes DDP
    broadcast + hook the *real* parameters; ``forward`` runs the compiled graph and
    returns ``(E, F)`` with the autograd graph intact, so a single
    ``loss.backward()`` drives DDP's bucketed all-reduce.  make_fx already baked the
    force 1st-backward into the graph, so this is an ordinary *single* backward —
    exactly the case DDP supports (the classic double-backward incompatibility is
    gone).
    """

    def __init__(self, makefx_step):
        super().__init__()
        self.model = makefx_step.model      # register the real params for DDP
        self._step = makefx_step

    def forward(self, batch):
        s = self._step
        # s._pvals ARE model.parameters() (same objects DDP hooks); DDP broadcasts
        # their data in-place at construction, so the captured tuple stays valid.
        E, F = s._gmc(s._pvals, *s._unpack(batch))
        return E, F


def _count_complex(gm):
    n = 0
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        v = node.meta.get("val", None)
        vals = v if isinstance(v, (list, tuple)) else [v]
        if any(isinstance(x, torch.Tensor) and x.is_complex() for x in vals):
            n += 1
    return n
