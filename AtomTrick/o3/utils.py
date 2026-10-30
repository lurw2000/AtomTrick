from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor


def _build_ch_info(idxs: List[int], slices: List[slice]):
    """Return channel access info for a group of entries.

    Returns a single ``slice`` if the channels are contiguous end-to-end
    (zero-copy view), otherwise a flat ``list[int]`` of channel indices
    (one gather, no intermediate cat).
    """
    if len(idxs) == 1:
        return slices[idxs[0]]
    s_list = [slices[i] for i in idxs]
    if all(s_list[k].stop == s_list[k + 1].start for k in range(len(s_list) - 1)):
        return slice(s_list[0].start, s_list[-1].stop)   # merged contiguous slice
    ch: List[int] = []
    for s in s_list:
        ch.extend(range(s.start, s.stop))
    return ch   # non-contiguous: flat index list


def _rebuild_o3tensor(data: Tensor, irreps):
    from .tensor import O3Tensor

    return O3Tensor(data, irreps)


# Pre-allocated constants for CUDA graph compatibility (no H2D during capture)
_rot_constants_cache: Dict[Tuple, Tuple[Tensor, Tensor, Tensor]] = {}


def _get_rot_constants(device, dtype):
    """Lazily create rotation constants on the correct device/dtype."""
    key = (device, dtype)
    if key not in _rot_constants_cache:
        _rot_constants_cache[key] = (
            torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype),
            torch.eye(3, device=device, dtype=dtype),
            torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype)),
        )
    return _rot_constants_cache[key]


def batched_rotation_to_z(directions: Tensor) -> Tuple[Tensor, Tensor]:
    """Proper rotations mapping each direction to +z, and their inverses (Rodrigues' formula).

    Parameters
    ----------
    directions : Tensor, shape ``(*, 3)``
        Unit direction vectors (must already be normalised).

    Returns
    -------
    R : Tensor, shape ``(*, 3, 3)``
        ``R[...]`` maps the corresponding direction onto ``[0, 0, 1]``.
    R_inv : Tensor, shape ``(*, 3, 3)``
        ``R_inv[...] = R[...].T`` (maps +z back to the direction).
    """
    batch_shape = directions.shape[:-1]
    d = directions.reshape(-1, 3)                         # (E, 3)

    z_unit, eye3, anti3 = _get_rot_constants(d.device, d.dtype)

    cos_a = d[:, 2]                                       # (E,)  dz = d·z
    dx, dy = d[:, 0], d[:, 1]
    # Stable "rotation between two unit vectors" formula (Moller-Hughes):
    #     v = d x z = [dy, -dx, 0],   R = I + [v]_x + [v]_x^2 / (1 + dz).
    # Algebraically identical to the Rodrigues form (sin_a*K + (1-cos_a)*K^2)
    # but built from the UN-normalised cross product, so there is NO 1/sin_a.
    # The old 1/sin_a construction was finite in the forward yet made the FORCE
    # gradient (dE/dd through the rotation) carry ~1/sin_a^3 terms that
    # catastrophically cancel in float32 for edges near the +z pole
    # (sin_a ~ 1e-3..1e-2, e.g. OC20 slab edges) -> spurious ~1e2 eV/A forces.
    # Here the +z pole is fully regular: 1/(1+dz) -> 1/2 and V -> 0 so R -> I
    # exactly, with a smooth O(1) gradient.  The ONLY remaining singularity is
    # the antipole dz = -1 (a 180 deg flip whose axis is genuinely undefined);
    # guard just that case as before (substitute a safe direction before the
    # 1/(1+dz), then overwrite R with anti3; grad wrt an antipole dir is 0).
    near_anti = cos_a < -1.0 + 1e-6                       # (E,) direction ≈ -z
    dx = torch.where(near_anti, torch.ones_like(dx), dx)
    dy = torch.where(near_anti, torch.zeros_like(dy), dy)
    cz = torch.where(near_anti, torch.zeros_like(cos_a), cos_a)
    zero = torch.zeros_like(dx)
    # V = [v]_x with v = d x z = [dy, -dx, 0]
    V = torch.stack([
        torch.stack([ zero, zero,  -dx], dim=-1),
        torch.stack([ zero, zero,  -dy], dim=-1),
        torch.stack([   dx,   dy, zero], dim=-1),
    ], dim=-2)  # (E, 3, 3)
    inv1p = 1.0 / (1.0 + cz)                              # (E,) regular at +z
    R = eye3.unsqueeze(0) + V + inv1p[:, None, None] * (V @ V)

    # Antipole override — branchless for CUDA graph compatibility.  (+z pole
    # needs no override: the formula already gives R = I there.)
    anti = near_anti.unsqueeze(-1).unsqueeze(-1)
    R = torch.where(anti, anti3, R)

    R = R.reshape(*batch_shape, 3, 3)
    return R, R.transpose(-1, -2)


# ---------------------------------------------------------------------------
# Non-persistent index buffers (+ loading checkpoints that still contain them)
# ---------------------------------------------------------------------------
#
# Every module in this package precomputes integer gather/scatter indices,
# masks and similar constants from its irreps at ``__init__``.  They are
# buffers (so ``.to(device)`` moves them and ``named_buffers()`` -- which the
# make_fx path feeds to ``functional_call`` -- sees them), but they carry no
# learned or data-derived information: the constructor rebuilds them exactly.
# Storing them in every checkpoint made a 3.7M-parameter model's ``best.pt``
# ~190 MB, so they are registered ``persistent=False``.
#
# Checkpoints written before that change still contain these keys, and a
# strict ``load_state_dict`` would reject them as "unexpected".  The pre-hook
# below drops them after checking that each saved copy equals the freshly built
# value, so old checkpoints -- including the pre-block-parameter flat-weight
# ones handled by ``linear._FlatWeightCompat`` -- keep loading with
# ``strict=True``.  A saved copy that DIFFERS means the index construction has
# changed since the checkpoint was written, i.e. the weights would silently
# compute a different function; that is reported as a load error.

def index_buffer_matches(saved: Tensor, current: Tensor) -> bool:
    """True if a checkpoint's copy of an index buffer equals the constructed one
    (same shape; integer/bool tables exactly, float constants to 1e-6)."""
    if tuple(saved.shape) != tuple(current.shape):
        return False
    saved = saved.detach().to(device=current.device)
    if current.is_floating_point() or saved.is_floating_point():
        return bool(torch.allclose(saved.to(torch.float64), current.to(torch.float64),
                                   rtol=1e-6, atol=1e-6))
    return bool(torch.equal(saved.to(current.dtype), current))


def _drop_legacy_index_buffer_keys(module, state_dict, prefix, local_metadata,
                                   strict, missing_keys, unexpected_keys,
                                   error_msgs):
    """load_state_dict pre-hook: check, then discard, saved copies of
    non-persistent index buffers."""
    for name in getattr(module, "_index_buffer_names", ()):
        saved = state_dict.pop(prefix + name, None)
        if saved is None:
            continue
        current = module._buffers.get(name)
        if current is not None and not index_buffer_matches(saved, current):
            error_msgs.append(
                f"saved index buffer {prefix + name} (shape {tuple(saved.shape)}) differs "
                f"from the one this code builds (shape {tuple(current.shape)}): the "
                "checkpoint was written by code with a different index layout and would "
                "compute a different function.  Load it with the code revision that "
                "wrote it.")


def register_index_buffer(module, name: str, tensor: Tensor) -> None:
    """``module.register_buffer(name, tensor, persistent=False)`` + legacy-load hook.

    Use ONLY for buffers whose value is fully reconstructed by the module's
    ``__init__`` from its configuration (index / mask / constant tables).
    Buffers that hold learned or data-derived values (e.g. the model's
    ``atomic_ref_energy`` and ``energy_bias``) must stay persistent.
    """
    module.register_buffer(name, tensor, persistent=False)
    names = module.__dict__.get("_index_buffer_names")
    if names is None:
        names = []
        module.__dict__["_index_buffer_names"] = names
        # with_module=True so the hook always acts on the module it is attached
        # to (also after copy.deepcopy, e.g. the EMA AveragedModel copy).
        module._register_load_state_dict_pre_hook(
            _drop_legacy_index_buffer_keys, with_module=True)
    if name not in names:
        names.append(name)
