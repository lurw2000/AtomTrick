"""Shared helpers for the baseline latency harnesses (stdlib + torch + numpy only).

Imported by ``baselines/<model>/benchmark_inference*.py``; it runs inside each baseline's own
environment (Python 3.10 - 3.13, torch 2.5.x), so it must stay dependency-free beyond torch /
numpy.  With default arguments each harness times exactly what the reported full-fp32 table
used; the other switches are listed in each harness docstring.

* ``--matmul-precision``: ``torch.set_float32_matmul_precision`` applied right before timing
  (after the model is built, so model code cannot silently override it).  ``highest`` (the
  PyTorch default, and what all baseline rows used) = strict fp32, no
  TF32; cuDNN TF32 is switched off with it.  The value in force while timing is written to
  every CSV row.
* ``no_create_graph()``: forces ``create_graph=False`` in every ``torch.autograd.grad`` call
  made while it is active.  Used for models whose upstream force code hard-codes
  ``create_graph=True`` (Equiformer, ``--create-graph false``, the default); at inference no
  second-order graph is needed, so this is the like-for-like setting (ATOMTRICK eager and MACE
  ``training=False`` already do it).
* ``load_raw_frames``: frames 0..n-1 of the raw rMD17 / MD22 downloads -- the inputs the MACE
  and GotenNet harnesses and ``bench_atomtrick.py --frame_source raw`` use.
* ``median_cuda_ms``: CUDA-event timing of an arbitrary callable (used to measure the
  neighbour-list cost that some models build inside their timed forward).
* ``--compile {none,inductor,fused}`` (``add_common_args``; default ``none`` = the eager
  protocol above, untouched): time the baseline under ``torch.compile(..., dynamic=True)``
  (TorchInductor, default mode, no CUDA graphs, ``fullgraph=False``) with two recipes.
  ``inductor``: the force ``autograd.grad`` keeps ``create_graph=False``; Dynamo graph-breaks
  at it, the energy forward is compiled and AOTAutograd's compiled backward computes the forces.
  ``fused``: ``torch._dynamo.allow_in_graph(torch.autograd.grad)`` with ``create_graph=True``
  (MACE's own ``prepare(allow_autograd=True)`` recipe, generalised), so the force backward is
  inlined into ONE compiled graph -- the analogue of ATOMTRICK's make_fx path.  The fused
  callable detaches its outputs INSIDE the compiled region (``fused_callable``), so AOTAutograd
  compiles an inference graph (no saved activations / autograd.Function wrapper for a backward
  that is never run) and executes it under no_grad, as ATOMTRICK's make_fx graph is; the
  inlined ``autograd.grad`` is traced with grad mode enabled either way.  Both recipes:
  ``PYTORCH_JIT=0`` before ``import torch`` (``pre_import_env``), e3nn's TorchScript codegen off
  (``e3nn_disable_script``), ``configure_dynamo`` (cache limits, no silent eager fallback),
  parameters frozen (``freeze_params``), positions ``requires_grad=True``, the whole timed
  callable is the compiled object, 10 warmup (= trace + compile, excluded) + 50 timed calls as
  for the eager rows.  ``CompileStats`` / ``compiled_timed_forwards`` record Dynamo's counters
  (graphs compiled during warmup / during the timed loop, graph breaks, warmup wall-clock) into
  the ``compile_*`` CSV columns; a row whose timed loop compiled is re-timed once
  (``compile_retimed=1``); after the warmup the compiled energy and forces are compared with
  one eager call of the same model on the same batch (``compare_outputs``; max relative error
  in ``compile_max_rel_err``, a mismatch or all-zero forces -> compile_error), and a Dynamo
  cache-limit fallback to eager (only a warning in torch) is turned into a compile_error
  (``DynamoCacheLimitWatch``).  This module therefore imports torch lazily (``pre_import_env``
  must run before torch is imported).
"""
import contextlib
import logging
import os
import statistics
import time

PROV_FIELDS = ["matmul_precision", "cudnn_allow_tf32", "param_dtype", "forces", "create_graph",
               "coords", "edges", "graph_build_median_ms", "torch_version", "cuda_version",
               # --compile provenance (empty for eager rows except compile=none / pytorch_jit)
               "compile", "compile_graphs_warmup", "compile_graphs_timed", "compile_graph_breaks",
               "compile_warmup_s", "compile_retimed", "compile_max_rel_err", "pytorch_jit"]

COMPILE_MODES = ("none", "inductor", "fused")
CHECK_SEED = 20261019   # RNG seed set before the compiled and the eager call of the output check

MD22_RAW = {
    "Ac_Ala3_NHMe": "md22_Ac-Ala3-NHMe.npz", "DHA": "md22_DHA.npz", "AT_AT": "md22_AT-AT.npz",
    "stachyose": "md22_stachyose.npz", "AT_AT_CG_CG": "md22_AT-AT-CG-CG.npz",
    "buckyball_catcher": "md22_buckyball-catcher.npz",
    "double_walled_nanotube": "md22_double-walled_nanotube.npz",
}


def add_common_args(ap):
    ap.add_argument("--matmul-precision", choices=["highest", "high", "medium"],
                    default="highest",
                    help="torch.set_float32_matmul_precision while timing (default highest = "
                         "strict fp32, no TF32; also sets cudnn.allow_tf32 accordingly)")
    ap.add_argument("--raw-root", default=os.environ.get("LATENCY_RAW_ROOT"),
                    help="directory holding raw/rmd17/npz_data/rmd17_<mol>.npz and "
                         "raw/md22/md22_<mol>.npz (for --coords raw; env LATENCY_RAW_ROOT)")
    ap.add_argument("--compile", choices=list(COMPILE_MODES), default="none",
                    help="none = eager (the reported protocol, default); inductor = "
                         "torch.compile(dynamic=True) of the whole timed callable, forces by "
                         "AOTAutograd's compiled backward (create_graph=False, graph break at "
                         "autograd.grad); fused = same plus allow_in_graph(autograd.grad) with "
                         "create_graph=True, i.e. forces inlined into ONE compiled graph (MACE's "
                         "upstream recipe).  Both set PYTORCH_JIT=0 before import torch and freeze "
                         "the parameters; compile time lands in the warmup calls.")


def compile_mode_from_argv(argv):
    """The value of --compile in ``argv`` (``--compile X`` or ``--compile=X``), else 'none'.
    Pure string scanning so it can run before torch is imported."""
    mode = "none"
    for i, a in enumerate(argv):
        if a == "--compile" and i + 1 < len(argv):
            mode = argv[i + 1]
        elif a.startswith("--compile="):
            mode = a.split("=", 1)[1]
    return mode


def pre_import_env(argv):
    """Call BEFORE ``import torch``: with ``--compile`` other than none, set ``PYTORCH_JIT=0``
    (unless already set) so TorchScript in e3nn / ``@torch.jit.script`` functions cannot block
    Dynamo.  Returns the compile mode found in argv.  No-op for the eager protocol."""
    mode = compile_mode_from_argv(argv)
    if mode != "none":
        os.environ.setdefault("PYTORCH_JIT", "0")
    return mode


def pytorch_jit_state():
    return os.environ.get("PYTORCH_JIT", "unset")


def compile_prov(mode):
    """The two compile columns every row carries (eager rows: compile=none)."""
    return {"compile": mode, "pytorch_jit": pytorch_jit_state()}


def e3nn_disable_script():
    """``e3nn.set_optimization_defaults(jit_script_fx=False)`` so e3nn's generated code stays
    plain Python / fx (traceable by Dynamo).  Call before the model is built.  True if e3nn is
    importable (and the default was set), False otherwise."""
    try:
        import e3nn
    except ImportError:
        return False
    e3nn.set_optimization_defaults(jit_script_fx=False)
    return True


def configure_dynamo():
    """Dynamo settings for the compiled rows: room for the dynamic-shape graphs of all molecules
    and batch sizes in one process, and NO silent eager fallback on a compile failure."""
    import torch
    import torch._dynamo
    cfg = torch._dynamo.config
    cfg.cache_size_limit = 64
    if hasattr(cfg, "accumulated_cache_size_limit"):
        cfg.accumulated_cache_size_limit = 1024
    cfg.suppress_errors = False
    # dynamic=True also turns plain Python ints that enter a frame as arguments or attributes
    # (channel counts, lmax, ...) into symbolic ints.  EquiformerV2 builds an empty
    # SO3_Embedding(0, ...) in its edge-degree embedding; with the 0 symbolic, Inductor 2.5
    # emits a kernel with size_hints=[0] and fails in triton_heuristics.pointwise ("reduce() of
    # empty iterable").  specialize_int=True keeps such ints constant (a static zero-size tensor
    # is handled); tensor dimensions stay dynamic, so one graph still serves every molecule.
    cfg.specialize_int = True
    # Data-dependent-shape ops (nonzero, boolean-mask indexing) stay graph breaks (the
    # default).  Capturing them (capture_dynamic_output_shape_ops=True) was tried for GotenNet
    # and makes its whole-model compile fail on an unbacked-size guard (Eq(u0, u7)); GotenNet's
    # real obstacle, two boolean-mask in-place assignments and an in-place unsqueeze_, is
    # handled by gotennet_compile.patch instead.
    # The inductor recipe runs AOTAutograd's compiled backward from torch.autograd.grad.  The
    # autograd engine launches CUDA backward nodes on its per-device worker thread, and in the
    # cu124 environments (GotenNet, Equiformer, EquiformerV2; torch 2.5.1+cu124) the Triton
    # launcher then fails with "Triton Error [CUDA]: invalid device context" (no CUDA context
    # current on that thread).  Single-threaded autograd runs the backward on the calling
    # thread, where the context is current.  Same work, same kernels; applied to every compiled
    # row for uniformity (the eager rows are untouched: the harness only calls this with
    # --compile != none).
    torch.autograd.set_multithreading_enabled(False)
    # EquiformerV2 draws a random vector in its forward (the gauge of every edge frame,
    # _init_edge_rot_mat), and with m_max < l_max its outputs depend on that draw.  Inductor's
    # own Philox RNG would then make the compiled call differ from the eager reference even
    # under the same seed, so the output check could not tell a wrong graph from a different
    # random draw.  fallback_random=True makes compiled RNG ops call the eager generator (same
    # numbers under the same seed; the check seeds both calls).  It touches only RNG ops.
    import torch._inductor.config as icfg
    icfg.fallback_random = True
    return {"cache_size_limit": cfg.cache_size_limit, "fallback_random": icfg.fallback_random,
            "accumulated_cache_size_limit": getattr(cfg, "accumulated_cache_size_limit", None),
            "suppress_errors": cfg.suppress_errors, "specialize_int": cfg.specialize_int,
            "autograd_multithreading": torch.autograd.is_multithreading_enabled(),
            "opaque_neighbor_ops": opaque_neighbor_ops()}


def opaque_neighbor_ops():
    """Make torch_cluster's neighbour-list functions opaque to Dynamo.

    ``torch_cluster.radius`` / ``radius_graph`` (used inside the forward of GotenNet, Equiformer
    and, through ``torch_geometric.nn.radius_graph``, EquiformerV2) are plain Python around the
    custom kernel ``torch.ops.torch_cluster.radius``, which has no FakeTensor (meta) kernel.
    Dynamo therefore traces INTO them and fails ("erroneously tracing into a custom kernel")
    instead of graph-breaking.  Wrapping them in ``torch._dynamo.disable`` turns that into a
    clean graph break: the neighbour list is built eagerly inside the timed region, exactly as
    in the eager rows, and the rest of the forward is compiled.  Patched in the defining
    submodule (``torch_cluster.radius`` looks ``radius`` up there at call time) and on the
    package, so it holds whichever name a model module imported.  Returns the patched names
    (empty when torch_cluster is not installed, e.g. the MACE environment)."""
    import sys
    try:
        import torch_cluster
    except ImportError:
        return []
    import torch._dynamo
    patched = []
    for modname, names in (("torch_cluster.radius", ("radius", "radius_graph")),
                           ("torch_cluster.knn", ("knn", "knn_graph"))):
        m = sys.modules.get(modname)
        if m is None:
            continue
        for n in names:
            f = getattr(m, n, None)
            if f is None or getattr(f, "_latency_opaque", False):
                continue
            w = torch._dynamo.disable(f)
            try:
                w._latency_opaque = True
            except AttributeError:
                pass
            setattr(m, n, w)
            if getattr(torch_cluster, n, None) is f:
                setattr(torch_cluster, n, w)
            patched.append(f"{modname}.{n}")
    return patched


def set_autograd_in_graph(allow):
    """fused: ``torch._dynamo.allow_in_graph(torch.autograd.grad)`` (the force backward is traced
    into the compiled graph; needs create_graph=True inside).  inductor: Dynamo's default rule
    for ``torch.autograd.grad`` is already a graph break (``SkipFunctionVariable`` in
    trace_rules), and ``torch._dynamo.disallow_in_graph`` RAISES ``IncorrectUsage`` on a callable
    that was never allowed (torch 2.5 decorators.py: "disallow_in_graph is expected to be used on
    an already allowed callable"), which is also why MACE's ``prepare(allow_autograd=False)``
    cannot be used; so ``disallow_in_graph`` is called only to undo a previous ``allow_in_graph``
    in the same process.  Returns the resulting state (True = allowed in graph).  Call before the
    model is compiled."""
    import torch
    import torch._dynamo
    fn = torch.autograd.grad
    if allow:
        torch._dynamo.allow_in_graph(fn)
        return True
    try:
        from torch._dynamo import trace_rules
        allowed = bool(trace_rules.is_callable_allowed(fn))
    except Exception:            # older/newer trace_rules: probe by trying to disallow
        allowed = None
    if allowed is False:
        return False
    try:
        torch._dynamo.disallow_in_graph(fn)
    except Exception as e:        # never allowed -> the default rule (graph break) stands
        if allowed:
            raise
        print(f"  dynamo: torch.autograd.grad keeps its default graph-break rule ({short_msg(e, 60)})",
              flush=True)
    return False


def freeze_params(model):
    """Inference deployment: no parameter gradients, otherwise AOTAutograd's compiled backward
    also computes them.  Positions keep requires_grad=True (set by the harness / model)."""
    model.requires_grad_(False)
    return model


def compile_callable(fn):
    """The compiled object = the whole timed callable (model forward incl. the force
    autograd.grad): TorchInductor, default mode, dynamic shapes, no CUDA graphs, graph breaks
    allowed (fullgraph=False)."""
    import torch
    return torch.compile(fn, dynamic=True)


def detach_tree(out):
    """``.detach()`` every tensor in a tensor / tuple / list / dict output (None and other
    values pass through)."""
    import torch
    if torch.is_tensor(out):
        return out.detach()
    if isinstance(out, dict):
        return {k: detach_tree(v) for k, v in out.items()}
    if isinstance(out, (list, tuple)):
        return type(out)(detach_tree(v) for v in out)
    return out


def fused_callable(fn):
    """The fused recipe's callable: ``fn`` (model forward incl. the inlined force autograd.grad,
    create_graph=True) with its outputs detached INSIDE the compiled region.  With the outputs
    not requiring grad AOTAutograd compiles an inference graph instead of a training (joint)
    graph that would save activations for a second-order backward that is never run, and runs
    it under no_grad (torch 2.5 runtime_wrappers.py: "When we have an inference graph, we run
    with grad disabled") -- the same shape as ATOMTRICK's make_fx step (traced with grad, run
    under no_grad).  The inlined autograd.grad is traced with grad mode enabled either way."""
    def fused(*args, **kwargs):
        return detach_tree(fn(*args, **kwargs))
    return fused


def compare_outputs(ref, out, rtol=1e-2):
    """Max relative error between the compiled (``out``) and eager (``ref``) energy / forces,
    given as matching tuples of tensors.  Raises RuntimeError when a shape differs, a value is
    not finite, a reference tensor is all zero (a traced ``autograd.grad`` that returned None
    -> MACE emits zeros) or max|out - ref| > rtol * max|ref|."""
    import torch
    worst = 0.0
    for i, (a, b) in enumerate(zip(ref, out)):
        a, b = a.detach().float(), b.detach().float()
        if a.shape != b.shape:
            raise RuntimeError(f"output {i}: compiled shape {tuple(b.shape)} != eager {tuple(a.shape)}")
        if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise RuntimeError(f"output {i}: non-finite values (eager finite="
                               f"{bool(torch.isfinite(a).all())}, compiled finite={bool(torch.isfinite(b).all())})")
        scale = float(a.abs().max())
        if scale == 0.0:
            raise RuntimeError(f"output {i}: eager reference is all zero")
        err = float((a - b).abs().max()) / scale
        worst = max(worst, err)
        if err > rtol:
            raise RuntimeError(f"output {i}: compiled vs eager max relative error {err:.3e} > {rtol:g}"
                               + (" (compiled output all zero)" if float(b.abs().max()) == 0.0 else ""))
    return worst


class DynamoCacheLimitWatch(logging.Handler):
    """Catches Dynamo's "torch._dynamo hit config.cache_size_limit / accumulated_cache_size_limit"
    warning (convert_frame.py): after it the frame is skipped and runs EAGERLY, with no exception
    and no counter change, which would make a compiled row indistinguishable from a valid one.
    ``compiled_timed_forwards`` turns a hit into a compile_error."""

    LOGGERS = ("torch._dynamo.convert_frame", "torch._dynamo")

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.hit = None

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        if "hit config." in msg and self.hit is None:
            self.hit = " ".join(msg.split())[:160]

    def __enter__(self):
        for name in self.LOGGERS:
            logging.getLogger(name).addHandler(self)
        return self

    def __exit__(self, *exc):
        for name in self.LOGGERS:
            logging.getLogger(name).removeHandler(self)
        return False

    def raise_if_hit(self, when):
        if self.hit is not None:
            raise RuntimeError(f"Dynamo cache limit hit during the {when} calls, the frame ran "
                               f"eagerly: {self.hit}")


CUDA_FATAL = ("illegal memory access", "device-side assert", "unspecified launch failure",
              "illegal instruction", "cuda error: launch failed")


def cuda_fatal(msg):
    """True for CUDA errors that leave the context unusable (every later CUDA call fails too):
    the harnesses then record the remaining rows as not run instead of dying mid-file."""
    m = msg.lower()
    return any(k in m for k in CUDA_FATAL)


def short_msg(e, n=100):
    """First ``n`` characters of ``str(e)`` with whitespace collapsed (Dynamo / Inductor error
    messages span many lines)."""
    return " ".join(str(e).split())[:n]


class CompileStats:
    """Snapshots of Dynamo's counters (``torch._dynamo.utils.counters``): unique graphs
    compiled, graph breaks, frames converted OK.  ``mark(name)`` stores a snapshot,
    ``delta(a, b)`` returns (graphs, breaks, frames) between two snapshots."""

    def __init__(self):
        from torch._dynamo.utils import counters
        self._c = counters
        self.snap = {}
        self._breaks_reported = False

    def read(self):
        c = self._c
        return (int(c["stats"]["unique_graphs"]), int(sum(c["graph_break"].values())),
                int(c["frames"]["ok"]))

    def mark(self, name):
        self.snap[name] = self.read()
        return self.snap[name]

    def delta(self, a, b):
        return tuple(y - x for x, y in zip(self.snap[a], self.snap[b]))

    def graph_breaks(self):
        return dict(self._c["graph_break"])

    def report_breaks_once(self):
        """Print the graph-break reasons (useful provenance) -- once per harness run, after the
        first compiled row."""
        if not self._breaks_reported:
            self._breaks_reported = True
            print(f"  dynamo graph breaks so far: {self.graph_breaks()}", flush=True)


def compiled_timed_forwards(call, warmup, iters, stats, eager_call=None, pick=None, rtol=1e-2):
    """Timing protocol of the harnesses' ``timed_forwards`` (warmup, synchronize,
    reset_peak_memory_stats, ``iters`` CUDA-event timed calls, synchronize, peak memory) for a
    compiled callable, plus the Dynamo-counter bookkeeping of the compiled rows.  Compile time
    (trace + Inductor) lands in the warmup calls, which are excluded from the latency.  If any
    graph was compiled DURING the timed loop the loop is run once more (``compile_retimed=1``,
    the second timing is kept; ``compile_graphs_timed`` reports the graphs compiled during the
    kept loop, so > 0 still flags the row).

    Output check (``eager_call`` + ``pick``): after the warmup, one compiled call and one call of
    the un-compiled model on the same batch are compared (``pick(out)`` -> tuple of tensors, e.g.
    (energy, forces); ``compare_outputs``); the max relative error is returned in
    ``compile_max_rel_err`` and a mismatch raises (-> compile_error).  If the eager reference
    runs out of memory the check is skipped (column empty, line printed).  A Dynamo
    cache-limit fallback to eager during the warmup or timed calls raises as well.
    Returns (times_ms, peak_bytes, info)."""
    import torch

    def timed_loop():
        torch.cuda.reset_peak_memory_stats()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            _ = call()
            ends[i].record()
        torch.cuda.synchronize()
        return ([s.elapsed_time(e) for s, e in zip(starts, ends)],
                torch.cuda.max_memory_allocated())

    with DynamoCacheLimitWatch() as watch:
        stats.mark("before_warmup")
        t0 = time.perf_counter()
        for _ in range(warmup):
            _ = call()
        torch.cuda.synchronize()
        warmup_s = time.perf_counter() - t0
        stats.mark("after_warmup")
        watch.raise_if_hit("warmup")

        rel_err = None
        if eager_call is not None:
            # same RNG state for both calls (a forward may draw random numbers, e.g.
            # EquiformerV2's edge-frame gauge; configure_dynamo sets fallback_random)
            torch.manual_seed(CHECK_SEED)
            out = pick(call()) if pick else call()
            try:
                torch.manual_seed(CHECK_SEED)
                ref = pick(eager_call()) if pick else eager_call()
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                print("  compile: output check SKIPPED (the eager reference call ran out of "
                      "memory)", flush=True)
                ref = None
                torch.cuda.empty_cache()
            if ref is not None:
                rel_err = compare_outputs(tuple(ref), tuple(out), rtol=rtol)
                print(f"  compile: outputs vs eager: max relative error {rel_err:.2e} "
                      f"(limit {rtol:g})", flush=True)
            del out, ref
            torch.cuda.synchronize()

        times_ms, peak_bytes = timed_loop()
        stats.mark("after_timed")
        graphs_warmup, breaks_warmup, _ = stats.delta("before_warmup", "after_warmup")
        graphs_timed = stats.delta("after_warmup", "after_timed")[0]
        retimed = 0
        if graphs_timed > 0:
            print(f"  compile: {graphs_timed} graph(s) compiled DURING the timed loop "
                  f"-- re-running the timed loop once", flush=True)
            retimed = 1
            times_ms, peak_bytes = timed_loop()
            stats.mark("after_retimed")
            graphs_timed = stats.delta("after_timed", "after_retimed")[0]
        watch.raise_if_hit("timed")
    info = {"compile_graphs_warmup": graphs_warmup, "compile_graphs_timed": graphs_timed,
            "compile_graph_breaks": breaks_warmup, "compile_warmup_s": round(warmup_s, 3),
            "compile_retimed": retimed,
            "compile_max_rel_err": None if rel_err is None else f"{rel_err:.3e}"}
    print(f"  compile: graphs warmup={graphs_warmup} timed={graphs_timed} "
          f"breaks={breaks_warmup} warmup={warmup_s:.1f}s retimed={retimed}", flush=True)
    return times_ms, peak_bytes, info


def set_precision(p):
    import torch
    torch.set_float32_matmul_precision(p)
    torch.backends.cudnn.allow_tf32 = (p != "highest")
    return precision_state()


def precision_state():
    import torch
    return {"matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda}


@contextlib.contextmanager
def no_create_graph(enabled=True, dynamo_disable=False):
    """Force create_graph=False (and hence retain_graph=False unless asked) in every
    torch.autograd.grad call made inside the block.  Counts the calls it rewrote.
    ``dynamo_disable=True`` (the --compile inductor rows) wraps the hook in
    ``torch._dynamo.disable`` so Dynamo graph-breaks cleanly at it instead of tracing the hook;
    enter the block ONCE for the whole compiled run (the hook's identity is guarded)."""
    import torch
    stats = {"calls": 0, "rewritten": 0}
    if not enabled:
        yield stats
        return
    orig = torch.autograd.grad

    def grad(*a, **k):
        stats["calls"] += 1
        if k.get("create_graph"):
            stats["rewritten"] += 1
        k["create_graph"] = False
        return orig(*a, **k)

    if dynamo_disable:
        import torch._dynamo
        grad = torch._dynamo.disable(grad)
    torch.autograd.grad = grad
    try:
        yield stats
    finally:
        torch.autograd.grad = orig


def load_raw_frames(raw_root, dataset, molecule, n):
    """Frames 0..n-1 (cyclic) of the raw download as a list of (z int64 (na,), pos fp32 (na,3))."""
    import numpy as np
    import torch
    if not raw_root:
        raise ValueError("--coords raw needs --raw-root (or env LATENCY_RAW_ROOT)")
    if dataset == "md22":
        d = np.load(os.path.join(raw_root, "raw", "md22", MD22_RAW[molecule]))
        z, R = d["z"], d["R"]
    else:
        d = np.load(os.path.join(raw_root, "raw", "rmd17", "npz_data", f"rmd17_{molecule}.npz"))
        z, R = d["nuclear_charges"], d["coords"]
    z = torch.from_numpy(np.asarray(z, dtype=np.int64))
    return [(z.clone(), torch.from_numpy(np.asarray(R[i % len(R)], dtype=np.float32)))
            for i in range(n)]


def median_cuda_ms(fn, warmup=10, iters=50):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        st[i].record(); fn(); en[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(st, en))


def param_dtype(model):
    dts = sorted({str(p.dtype).replace("torch.", "") for p in model.parameters()})
    return "/".join(dts)
