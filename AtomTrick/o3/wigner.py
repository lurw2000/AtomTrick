"""Wigner D-matrix utilities for O(3) rotations and reflections.

All functions are implemented in pure PyTorch for GPU support and
automatic differentiation.  Euler angles use the ZYZ convention:

    R(α, β, γ) = Rz(α) · Ry(β) · Rz(γ)

The Wigner D-matrix of degree *l* transforms the 2l+1 spherical-harmonic
components under rotation::

    D^l(R) · f^l  =  rotated f^l

where rows/columns are indexed by m = -l, -l+1, ..., l.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor


# ── small wigner-d matrix (rotation about y-axis) ──────────────────────


def wigner_d(l: int, beta: Tensor) -> Tensor:
    """Small Wigner d-matrix  d^l_{mm'}(β).

    Uses the standard trigonometric formula (Varshalovich §4.3.1)::

        d^l_{mm'}(β) = Σ_s  (-1)^{m-m'+s}
            * √[(l+m)!(l-m)!(l+m')!(l-m')!]
            / [(l+m-s)! s! (m'-m+s)! (l-m'-s)!]
            * cos(β/2)^{2l-2s+m-m'}  sin(β/2)^{2s+m'-m}

    Parameters
    ----------
    l : int
        Degree (non-negative integer).
    beta : Tensor
        Rotation angle(s) about y-axis.  Arbitrary shape; the returned
        matrix has shape ``(*beta.shape, 2l+1, 2l+1)``.

    Returns
    -------
    Tensor
        Shape ``(*beta.shape, 2l+1, 2l+1)``.
    """
    if l == 0:
        return torch.ones(*beta.shape, 1, 1, dtype=beta.dtype, device=beta.device)

    dim = 2 * l + 1
    # precompute log-factorials: max argument across a,b,c,d is 4l
    # (c = m'-m+s can reach l-(-l)+2l = 4l)
    log_fact = _log_factorial_table(4 * l, dtype=beta.dtype, device=beta.device)

    cos_half = torch.cos(beta / 2)  # (*shape,)
    sin_half = torch.sin(beta / 2)

    # add trailing dims for (m, m') broadcasting
    cos_half = cos_half[..., None, None]  # (*shape, 1, 1)
    sin_half = sin_half[..., None, None]

    ms = torch.arange(-l, l + 1, device=beta.device)  # (dim,)
    m1 = ms[None, :]   # (1, dim) — row index m
    m2 = ms[:, None]    # (dim, 1) — col index m'

    # log prefactor: 0.5 * [log(l+m)!+log(l-m)!+log(l+m')!+log(l-m')!]
    log_pre = 0.5 * (
        log_fact[l + m1] + log_fact[l - m1]
        + log_fact[l + m2] + log_fact[l - m2]
    )  # (dim, dim)

    # summation over valid s
    s_max = 2 * l
    s_vals = torch.arange(s_max + 1, device=beta.device)  # (S,)

    # validity masks for each (m, m', s) triple
    # conditions: l+m-s >= 0, s >= 0, m'-m+s >= 0, l-m'-s >= 0
    a = (l + m1).unsqueeze(-1) - s_vals  # (dim, dim, S)  = l+m-s
    b = s_vals.unsqueeze(0).unsqueeze(0).expand(dim, dim, -1)  # s
    c = (m2 - m1).unsqueeze(-1) + s_vals  # m'-m+s
    d = (l - m2).unsqueeze(-1) - s_vals    # l-m'-s

    valid = (a >= 0) & (c >= 0) & (d >= 0)  # (dim, dim, S)

    # log denominator: log(a!) + log(b!) + log(c!) + log(d!)
    # clamp to 0 for invalid entries (won't matter, masked out)
    _lim = 4 * l  # upper clamp to stay within table bounds
    log_denom = (
        log_fact[a.clamp(min=0, max=_lim)]
        + log_fact[b]
        + log_fact[c.clamp(min=0, max=_lim)]
        + log_fact[d.clamp(min=0, max=_lim)]
    )  # (dim, dim, S)

    # exponents of cos(β/2) and sin(β/2)
    pow_cos = (2 * l - 2 * s_vals + (m1 - m2).unsqueeze(-1)).float()  # (dim, dim, S)
    pow_sin = (2 * s_vals + (m2 - m1).unsqueeze(-1)).float()

    # sign: (-1)^{m - m' + s}
    sign_exp = ((m1 - m2).unsqueeze(-1) + s_vals)
    sign = torch.where(sign_exp % 2 == 0, 1.0, -1.0)  # (dim, dim, S)

    # put it all together in log-space, then exponentiate
    # term = sign * exp(log_pre - log_denom) * cos^pow_cos * sin^pow_sin
    log_cos = torch.log(cos_half.abs().clamp(min=1e-45))  # (*shape, 1, 1)
    log_sin = torch.log(sin_half.abs().clamp(min=1e-45))
    sign_cos = torch.sign(cos_half)
    sign_sin = torch.sign(sin_half)

    # (*shape, dim, dim, S)
    log_term = (
        log_pre.unsqueeze(-1)
        - log_denom
        + pow_cos * log_cos.unsqueeze(-1)
        + pow_sin * log_sin.unsqueeze(-1)
    )
    # handle sign of cos^a * sin^b
    trig_sign = _pow_sign(sign_cos.unsqueeze(-1), pow_cos) * _pow_sign(
        sign_sin.unsqueeze(-1), pow_sin
    )
    terms = sign * trig_sign * torch.exp(log_term)
    terms = torch.where(valid, terms, torch.zeros_like(terms))

    return terms.sum(dim=-1)  # (*shape, dim, dim)


# ── full Wigner D-matrix from ZYZ Euler angles ─────────────────────────


_wigner_D_module_cache: Dict[Tuple[int, torch.dtype, torch.device], "WignerDSmall"] = {}


def _get_wigner_D_module(l: int, dtype, device) -> "WignerDSmall":
    """Module-level cache of :class:`WignerDSmall` — one instance per
    (l, dtype, device).  Avoids rebuilding the per-l tables on every call."""
    key = (l, dtype, device)
    mod = _wigner_D_module_cache.get(key)
    if mod is None:
        mod = WignerDSmall(l, dtype=dtype, device=device)
        _wigner_D_module_cache[key] = mod
    return mod


def wigner_D(l: int, alpha: Tensor, beta: Tensor, gamma: Tensor) -> Tensor:
    """Wigner D-matrix  D^l_{mm'}(α, β, γ)  in the ZYZ convention.

    Dispatches to a module-level cached :class:`WignerDSmall` so all
    per-l constants (log-factorial table, sign array, validity mask,
    change-of-basis ``U``) are computed once and reused across calls.
    The scripted core uses mask-before-exp so autograd propagates
    correctly through the Varshalovich log-space formula — verified
    to match the inline formula's gradients to machine epsilon at FP64
    (see ``benchmark/test_wigner_small_fixed.py``).

    Gives a ~3.5× forward speedup over the inline version at typical
    edge-batch sizes, with bit-identical forward values to FP32 rounding
    and FP64-exact gradients.

    Parameters
    ----------
    l : int
        Degree (non-negative).
    alpha, beta, gamma : Tensor
        ZYZ Euler angles.  Must be broadcastable to a common shape.
        The returned matrix has shape ``(*shape, 2l+1, 2l+1)``.

    Returns
    -------
    Tensor (real)
        Shape ``(*shape, 2l+1, 2l+1)``.
    """
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    if l == 0:
        return torch.ones(
            *alpha.shape, 1, 1, dtype=alpha.dtype, device=alpha.device
        )
    mod = _get_wigner_D_module(l, alpha.dtype, alpha.device)
    return mod(alpha, beta, gamma)


# ── rotation matrix ↔ Euler angles (ZYZ) ───────────────────────────────


def matrix_to_euler_ZYZ(R: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert 3×3 rotation matrix to ZYZ Euler angles (α, β, γ).

    Parameters
    ----------
    R : Tensor
        Rotation matrix, shape ``(*batch, 3, 3)``.

    Returns
    -------
    alpha, beta, gamma : Tensor
        Each with shape ``(*batch,)``.
    """
    # β = arccos(R_33).  acos'(±1) = -inf, so clamp the INPUT strictly inside
    # (-1, 1): at gimbal lock (OC20's axis-aligned edges give R_33=±1 exactly)
    # the clamp saturates → finite (zero) gradient instead of NaN.  NOTE the
    # clamp is on the INPUT: because acos'(1) = -inf it puts β at 4.472e-04,
    # NOT 1e-7, so β is restored exactly below (see the degen branch).
    r22 = R[..., 2, 2]
    beta = torch.acos(r22.clamp(-1.0 + 1e-7, 1.0 - 1e-7))

    # Degenerate cases — branchless for CUDA graph compatibility.  Detect from
    # the RAW R_33, not the clamped β: axis-aligned edges have R_33=±1 exactly,
    # which the clamp pushes to β≈4e-4 and would misclassify as non-degenerate,
    # letting their atan2 2nd-derivative (~1/sin²β) overflow in the double
    # backward (force-matching training).  R_33 is exact.
    tiny = 1e-6
    degen = r22 > 1.0 - tiny                      # β ≈ 0   (R ≈ identity)
    degen_pi = r22 < -1.0 + tiny                  # β ≈ π   (R ≈ anti-z)
    nondegen = ~(degen | degen_pi)

    # Restore beta EXACTLY at the poles.  The clamp above exists so acos' stays
    # finite, but acos'(1) = -inf means a 1e-7 clamp on the INPUT lands at
    # acos(1 - 1e-7) = 4.472e-04 on BETA -- not 1e-7.  Leaving it there makes D
    # wrong by ~1e-3 for any axis-aligned rotation: d^l_00 picks up
    # 1 - l(l+1)beta^2/4 (measured 0.9999999 / 0.9999997 / 0.9999994 for
    # l = 1/2/3), the |m| blocks leak into each other (6.3e-4 / 1.4e-3 / 2.4e-3),
    # and the representation law fails at the same order
    # (||D(2th) - D(th)^2|| ~ 3e-4..8e-4).  All of that survives in float64, so
    # it is a parametrisation error, not a precision one.
    #
    # degen / degen_pi are detected from the RAW R_33, so this is exact at the
    # poles and untouched elsewhere.  torch.where masks the discarded acos
    # branch, so gradients stay finite -- the property the clamp was added for.
    beta = torch.where(degen, torch.zeros_like(beta), beta)
    beta = torch.where(degen_pi, torch.full_like(beta, math.pi), beta)

    # Every atan2 below is a where-branch used on only SOME elements but is
    # COMPUTED for all — and atan2(0,0) has a NaN gradient that torch.where
    # leaks via 0*NaN.  Mask each atan2's inputs so, wherever its branch is
    # discarded, it sees a safe (0, 1) → atan2 = 0 with a finite gradient.
    # Values are unchanged from the original for non-degenerate R.
    one = torch.ones_like(r22); zero = torch.zeros_like(r22)
    a_y = torch.where(nondegen, R[..., 1, 2], zero)      # α = atan2(R_23, R_13)
    a_x = torch.where(nondegen, R[..., 0, 2], one)
    g_y = torch.where(nondegen, R[..., 2, 1], zero)      # γ = atan2(R_32, -R_31)
    g_x = torch.where(nondegen, -R[..., 2, 0], one)
    alpha = torch.atan2(a_y, a_x)
    gamma = torch.atan2(g_y, g_x)

    d0_y = torch.where(degen, R[..., 1, 0], zero)
    d0_x = torch.where(degen, R[..., 0, 0], one)
    alpha = torch.where(degen, torch.atan2(d0_y, d0_x), alpha)
    gamma = torch.where(degen, zero, gamma)

    # β = π: R = Rz(α) Ry(π) (γ folded into α) = [[-cos α, -sin α, 0],
    # [-sin α, cos α, 0], [0, 0, -1]], so α = atan2(-R_21, -R_11).  (The model's
    # eSCN frame never reaches this branch: its R_33 > -1 by construction.)
    dp_y = torch.where(degen_pi, -R[..., 1, 0], zero)
    dp_x = torch.where(degen_pi, -R[..., 0, 0], one)
    alpha = torch.where(degen_pi, torch.atan2(dp_y, dp_x), alpha)
    gamma = torch.where(degen_pi, zero, gamma)

    return alpha, beta, gamma


def euler_ZYZ_to_matrix(
    alpha: Tensor, beta: Tensor, gamma: Tensor
) -> Tensor:
    """Convert ZYZ Euler angles to a 3×3 rotation matrix.

    R(α, β, γ) = Rz(α) · Ry(β) · Rz(γ)

    Parameters
    ----------
    alpha, beta, gamma : Tensor
        ZYZ Euler angles; must be broadcastable.

    Returns
    -------
    Tensor
        Shape ``(*batch, 3, 3)``.
    """
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    cb, sb = torch.cos(beta), torch.sin(beta)
    cg, sg = torch.cos(gamma), torch.sin(gamma)

    # R = Rz(α) Ry(β) Rz(γ)
    R = torch.stack(
        [
            ca * cb * cg - sa * sg,
            -ca * cb * sg - sa * cg,
            ca * sb,
            sa * cb * cg + ca * sg,
            -sa * cb * sg + ca * cg,
            sa * sb,
            -sb * cg,
            sb * sg,
            cb,
        ],
        dim=-1,
    ).reshape(*alpha.shape, 3, 3)
    return R


# ── convenience: D-matrix from a rotation matrix ───────────────────────


def wigner_D_from_matrix(l: int, R: Tensor) -> Tensor:
    """Wigner D-matrix for degree *l* from a 3×3 rotation matrix.

    Parameters
    ----------
    l : int
        Degree.
    R : Tensor
        Rotation matrix, shape ``(*batch, 3, 3)``.

    Returns
    -------
    Tensor
        Shape ``(*batch, 2l+1, 2l+1)``.
    """
    alpha, beta, gamma = matrix_to_euler_ZYZ(R)
    return wigner_D(l, alpha, beta, gamma)


# ── WignerDSmall: per-l precomputed tables + scripted evaluator ────────


@torch.jit.script
def _wigner_d_small_real_core(
    beta: Tensor,
    log_pre: Tensor,        # (dim, dim)
    log_denom: Tensor,      # (dim, dim, S)
    pow_cos: Tensor,        # (dim, dim, S)  float
    pow_sin: Tensor,        # (dim, dim, S)  float
    sign_arr: Tensor,       # (dim, dim, S)
    valid_mask: Tensor,     # (dim, dim, S)
) -> Tensor:
    """Real-valued small Wigner-d(β) via the log-factorial formula.

    This is the scripted part — *no* complex tensors (TorchScript's
    complex autograd is flaky).  Returns the (*batch, dim, dim) real
    matrix d^l(β); the complex phase composition with α and γ happens
    in regular Python code where complex autograd is well-supported.
    """
    cos_half = torch.cos(beta / 2)
    sin_half = torch.sin(beta / 2)
    ch = cos_half.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    sh = sin_half.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    log_c = torch.log(ch.abs().clamp(min=1e-45))
    log_s = torch.log(sh.abs().clamp(min=1e-45))
    sign_c = torch.sign(ch)

    log_term = (
        log_pre.unsqueeze(-1)
        - log_denom
        + pow_cos * log_c
        + pow_sin * log_s
    )
    # Mask INVALID entries BEFORE the exp — avoids inf * 0 → NaN in backward.
    safe_log_term = torch.where(valid_mask, log_term, torch.zeros_like(log_term))

    pow_cos_odd = (pow_cos.long() % 2) != 0
    pow_sin_odd = (pow_sin.long() % 2) != 0
    neg_c = sign_c < 0
    neg_s = sh < 0
    trig_sign_c = torch.where(
        neg_c & pow_cos_odd, -torch.ones_like(ch), torch.ones_like(ch)
    )
    trig_sign_s = torch.where(
        neg_s & pow_sin_odd, -torch.ones_like(sh), torch.ones_like(sh)
    )
    trig_sign = trig_sign_c * trig_sign_s

    terms = sign_arr * trig_sign * torch.exp(safe_log_term)
    # Multiplicative mask — NaN-safe through backward.
    terms = terms * valid_mask.to(terms.dtype)
    return terms.sum(dim=-1)


def _wigner_D_small_core(
    alpha: Tensor,
    beta: Tensor,
    gamma: Tensor,
    log_pre: Tensor,        # (dim, dim)
    log_denom: Tensor,      # (dim, dim, S)
    pow_cos: Tensor,        # (dim, dim, S)  float
    pow_sin: Tensor,        # (dim, dim, S)  float
    sign_arr: Tensor,       # (dim, dim, S)
    valid_mask: Tensor,     # (dim, dim, S)
    ms: Tensor,             # (dim,)  -l..+l
    U_re: Tensor,           # (dim, dim)  Re(U)  complex->real change of basis
    U_im: Tensor,           # (dim, dim)  Im(U)
) -> Tensor:
    """Real-basis Wigner-D via a scripted real-only d(β) core plus a fully
    REAL phase / change-of-basis composition — NO complex tensors.

    Bit-identical to the former complex path ``Re(U · P_α d P_γ · U^†)`` but
    expanded in real arithmetic so make_fx / inductor can trace and compile
    the 2nd backward (complex conjugate-views raised 'view not supported for
    conjugate view tensors when converting to a different dtype' under
    AOTAutograd, blocking the standard-API force-matching training path).

    Derivation: the complex Wigner-D is
        D_c[m,m'] = d[m,m'] · exp(-i(m α + m' γ)) = d·cos φ − i·d·sin φ,
        φ[m,m'] = ms[m]·α + ms[m']·γ.
    With U = U_re + i·U_im and D_real = Re(U D_c U^†):
        M = U D_c :  Re(M) = U_re·dc_re − U_im·dc_im,
                     Im(M) = U_re·dc_im + U_im·dc_re
        D_real = Re(M)·U_reᵀ + Im(M)·U_imᵀ.
    """
    # Scripted real part: d^l(β).  (*batch, dim, dim) real.
    d = _wigner_d_small_real_core(
        beta, log_pre, log_denom, pow_cos, pow_sin, sign_arr, valid_mask,
    )

    # φ[...,m,m'] = ms[m]·α + ms[m']·γ   (real; *batch, dim, dim)
    phi = (ms.unsqueeze(-1) * alpha.unsqueeze(-1).unsqueeze(-1)
           + ms.unsqueeze(-2) * gamma.unsqueeze(-1).unsqueeze(-1))
    dc_re = d * torch.cos(phi)
    dc_im = -d * torch.sin(phi)

    # D_real = Re(U (dc_re + i dc_im) U^†) — all real matmuls with constant U_re/U_im.
    m_re = torch.matmul(U_re, dc_re) - torch.matmul(U_im, dc_im)
    m_im = torch.matmul(U_re, dc_im) + torch.matmul(U_im, dc_re)
    D_real = (torch.matmul(m_re, U_re.transpose(-2, -1))
              + torch.matmul(m_im, U_im.transpose(-2, -1)))
    return D_real.to(alpha.dtype)


class WignerDSmall(torch.nn.Module):
    """Precomputed fast Wigner-D evaluator for a fixed set of degrees.

    Bakes every l-dependent constant (log-factorial prefactor, validity
    mask, pow-cos/sin exponents, sign array, real-basis change-of-basis
    matrix U) as buffers at ``__init__``.  The forward is a single
    ``@torch.jit.script`` call that evaluates ``D^l_real(α, β, γ)``
    given just the three angles.

    Numerically **identical** to :func:`wigner_D` at FP64 — verified in
    ``benchmark/test_wigner_small_equiv.py`` with residual < 1e-13
    across 10⁴ random angles for ``l ≤ 6``.

    Typical usage::

        wig = WignerDSmall(l=3, dtype=torch.float32, device='cuda')
        D = wig(alpha, beta, gamma)   # (*batch, 2l+1, 2l+1)
    """

    def __init__(self, l: int, dtype=torch.float32, device=None):
        super().__init__()
        device = device or torch.device("cpu")
        self.l = l
        dim = 2 * l + 1

        # Build all per-l constants in the requested dtype — fp32 by
        # default (the project's runtime precision).  The log-factorial
        # table has at most 4·l+1 ≤ 25 entries for l ≤ 6, so fp32 init
        # carries < 1 ULP accumulated error; we verified fp32 buffers
        # give the same Wigner-D to < 1e-5 vs fp64 buffers in
        # ``benchmark/test_wigner_fp32_precision.py``.  If tighter init
        # precision is ever required, construct with ``dtype=torch.float64``.
        work_dtype = dtype

        ms = torch.arange(-l, l + 1, dtype=work_dtype, device=device)
        m1 = ms[None, :]                      # (1, dim)   row index m
        m2 = ms[:, None]                      # (dim, 1)   col index m'

        log_fact = _log_factorial_table(4 * l, dtype=work_dtype, device=device)

        log_pre = 0.5 * (
            log_fact[(l + m1).long()] + log_fact[(l - m1).long()]
            + log_fact[(l + m2).long()] + log_fact[(l - m2).long()]
        )                                      # (dim, dim)

        s_max = 2 * l
        s_vals = torch.arange(s_max + 1, dtype=work_dtype, device=device)

        a = (l + m1).unsqueeze(-1) - s_vals    # (dim, dim, S)
        c = (m2 - m1).unsqueeze(-1) + s_vals
        d_idx = (l - m2).unsqueeze(-1) - s_vals
        b = s_vals.expand(dim, dim, -1).contiguous()

        valid = (a >= 0) & (c >= 0) & (d_idx >= 0)

        _lim = 4 * l
        log_denom = (
            log_fact[a.clamp(min=0, max=_lim).long()]
            + log_fact[b.long()]
            + log_fact[c.clamp(min=0, max=_lim).long()]
            + log_fact[d_idx.clamp(min=0, max=_lim).long()]
        )

        pow_cos = (2 * l - 2 * s_vals + (m1 - m2).unsqueeze(-1))
        pow_sin = (2 * s_vals + (m2 - m1).unsqueeze(-1))

        sign_exp = ((m1 - m2).unsqueeze(-1) + s_vals).long()
        sign = torch.where(
            sign_exp % 2 == 0,
            torch.ones_like(pow_cos),
            -torch.ones_like(pow_cos),
        )

        U = _change_of_basis_real_complex(l, dtype=work_dtype, device=device)

        # cast to runtime dtype
        self.register_buffer("_log_pre", log_pre.to(dtype))
        self.register_buffer("_log_denom", log_denom.to(dtype))
        self.register_buffer("_pow_cos", pow_cos.to(dtype))
        self.register_buffer("_pow_sin", pow_sin.to(dtype))
        self.register_buffer("_sign", sign.to(dtype))
        self.register_buffer("_valid", valid)
        self.register_buffer("_ms", ms.to(dtype))
        # Store the REAL and IMAG parts of the complex->real change-of-basis so
        # the forward composition is fully real (no complex tensors -> the
        # make_fx / inductor 2nd backward can compile; see _wigner_D_small_core).
        self.register_buffer("_U_re", U.real.to(dtype).contiguous())
        self.register_buffer("_U_im", U.imag.to(dtype).contiguous())

    def forward(self, alpha: Tensor, beta: Tensor, gamma: Tensor) -> Tensor:
        if self.l == 0:
            return torch.ones(
                *alpha.shape, 1, 1, dtype=alpha.dtype, device=alpha.device
            )
        alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
        return _wigner_D_small_core(
            alpha, beta, gamma,
            self._log_pre, self._log_denom, self._pow_cos, self._pow_sin,
            self._sign, self._valid, self._ms, self._U_re, self._U_im,
        )


# ── pre-computed D-matrix cache for a set of rotations ────────────────


def precompute_wigner_D(
    ls: Union[List[int], set],
    alpha: Tensor,
    beta: Tensor,
    gamma: Tensor,
) -> Dict[int, Tensor]:
    """Pre-compute Wigner D-matrices for several degrees at once.

    Returns a dict ``{l: D^l}`` where each ``D^l`` has shape
    ``(*batch, 2l+1, 2l+1)``.  Useful when the *same* rotation is
    applied to multiple irrep blocks (e.g. in ``rotate_irreps``).

    Parameters
    ----------
    ls : list or set of int
        Degrees to compute.
    alpha, beta, gamma : Tensor
        ZYZ Euler angles (broadcastable).

    Returns
    -------
    dict mapping int → Tensor
    """
    cache: Dict[int, Tensor] = {}
    for l in ls:
        cache[l] = wigner_D(l, alpha, beta, gamma)
    return cache


def precompute_wigner_D_from_matrix(
    ls: Union[List[int], set],
    R: Tensor,
    *,
    parity: Optional[Tensor] = None,
) -> Tuple[Dict[int, Tensor], Tensor]:
    """Pre-compute Wigner D-matrices from a rotation matrix.

    Like :func:`precompute_wigner_D` but takes a 3×3 rotation matrix
    and also returns the parity (``det(R).sign()``).

    Parameters
    ----------
    ls : list or set of int
        Degrees to compute.
    R : Tensor
        Rotation matrix, shape ``(*batch, 3, 3)``.
    parity : Tensor, optional
        Determinant sign; computed if ``None``.

    Returns
    -------
    (dict mapping int → Tensor, parity Tensor)
    """
    if parity is None:
        parity = _det3x3(R).sign()
    R_proper = parity[..., None, None] * R
    alpha, beta, gamma = matrix_to_euler_ZYZ(R_proper)
    cache = precompute_wigner_D(ls, alpha, beta, gamma)
    return cache, parity


# ── block-diagonal D-matrix for an Irreps layout ───────────────────────


def wigner_D_block(
    irreps,
    R: Tensor,
    *,
    parity: Optional[Tensor] = None,
) -> Tensor:
    """Block-diagonal Wigner D-matrix matching an :class:`Irreps` layout.

    For each ``(mult, Irrep(l, p))`` entry the block is
    ``I_mult ⊗ D^l(R)`` (optionally multiplied by ``p * det`` when the
    irrep has odd parity, to handle improper rotations / reflections).

    Parameters
    ----------
    irreps : Irreps
        The irrep layout.
    R : Tensor
        Rotation matrix, shape ``(*batch, 3, 3)``.
    parity : Tensor, optional
        Determinant of R (``+1`` or ``-1``), shape ``(*batch,)``.
        If ``None``, computed automatically as ``det(R).sign()``.

    Returns
    -------
    Tensor
        Shape ``(*batch, irreps.dim, irreps.dim)``, block-diagonal.
    """
    if parity is None:
        parity = _det3x3(R).sign()

    # Extract the proper rotation: for improper R (det=-1), the proper
    # rotation part is det(R)*R.  Euler decomposition only works on SO(3).
    R_proper = parity[..., None, None] * R
    alpha, beta, gamma = matrix_to_euler_ZYZ(R_proper)
    batch = alpha.shape

    blocks: List[Tensor] = []
    for mult, ir in irreps:
        D = wigner_D(ir.l, alpha, beta, gamma)  # (*batch, 2l+1, 2l+1)
        # for odd-parity irreps, multiply by det(R) = parity
        if ir.p == -1:
            D = D * parity[..., None, None]
        # Kronecker product with I_mult: I_mult ⊗ D^l
        if mult == 1:
            blocks.append(D)
        else:
            eye = torch.eye(mult, dtype=D.dtype, device=D.device)
            # kron(I_mult, D) via einsum: (mult, mult) x (*batch, dim, dim)
            #   → (*batch, mult*dim, mult*dim)
            block = torch.einsum("ij,...kl->...ikjl", eye, D)
            block = block.reshape(*batch, mult * ir.dim, mult * ir.dim)
            blocks.append(block)

    return _batched_block_diag(blocks, batch)


@torch.jit.script
def _rotate_irreps_core(
    x: Tensor,
    D_list: List[Tensor],
    parity: Tensor,
    starts: List[int],
    mults: List[int],
    dims: List[int],
    apply_parity: List[bool],
) -> Tensor:
    """Scripted hot loop: apply per-sector Wigner D blocks to ``x``.

    All per-irrep metadata is pre-resolved as parallel lists so the
    TorchScript loop has no Python object access.  Each step does one
    ``reshape`` + one ``matmul`` per sector.
    """
    parts: List[Tensor] = []
    batch_shape = x.shape[:-1]
    for i in range(len(D_list)):
        D = D_list[i]
        if apply_parity[i]:
            # broadcast parity (*batch,) to (*batch, 1, 1)
            D = D * parity.unsqueeze(-1).unsqueeze(-1)
        start = starts[i]
        mult = mults[i]
        dim = dims[i]
        x_slice = x.narrow(-1, start, mult * dim)
        # (*batch, mult, dim)
        x_block = x_slice.reshape(batch_shape + (mult, dim))
        # D: (*batch, dim, dim); want out[m, i] = sum_j D[i, j] * x_block[m, j]
        # = einsum("...ij,...mj->...mi", D, x_block)
        # equivalent to x_block @ D.transpose(-2,-1)
        out_block = torch.matmul(x_block, D.transpose(-2, -1))
        parts.append(out_block.reshape(batch_shape + (mult * dim,)))
    return torch.cat(parts, dim=-1)


def rotate_irreps(
    irreps,
    R: Tensor,
    x: Tensor,
    *,
    parity: Optional[Tensor] = None,
    D_cache: Optional[Dict[int, Tensor]] = None,
) -> Tensor:
    """Apply block-diagonal Wigner-D rotation to a feature vector, block-by-block.

    Mathematically equivalent to ``wigner_D_block(irreps, R) @ x`` but
    **never materialises the full (D, D) matrix**, operating on each
    ``(mult, l)`` block independently.  This reduces peak memory from
    O(D²) to O(max_block²) per sample.

    The per-sector loop is dispatched to a ``@torch.jit.script`` core for
    kernel-launch amortisation and ``torch.compile`` friendliness.

    Parameters
    ----------
    irreps : Irreps
        The irrep layout of *x*.
    R : Tensor
        Rotation matrix, shape ``(*batch, 3, 3)``.  Ignored when
        ``D_cache`` is provided.
    x : Tensor
        Feature vector, shape ``(*batch, irreps.dim)``.
    parity : Tensor, optional
        Determinant of *R* (``+1`` or ``-1``), shape ``(*batch,)``.
        Computed automatically when ``None``.
    D_cache : dict mapping int → Tensor, optional
        Pre-computed Wigner D-matrices keyed by degree *l*.  When
        provided, the Euler decomposition and per-*l* ``wigner_D``
        calls are skipped entirely.  Obtain via
        :func:`precompute_wigner_D_from_matrix`.

    Returns
    -------
    Tensor
        Rotated feature vector, same shape as *x*.
    """
    if parity is None:
        parity = _det3x3(R).sign()

    if D_cache is None:
        R_proper = parity[..., None, None] * R
        alpha, beta, gamma = matrix_to_euler_ZYZ(R_proper)

    # Resolve per-sector metadata and D tensors once.
    D_list: List[Tensor] = []
    starts: List[int] = []
    mults: List[int] = []
    dims: List[int] = []
    apply_parity: List[bool] = []
    for (mult, ir), sl in zip(irreps, irreps.slices()):
        if D_cache is not None and ir.l in D_cache:
            D = D_cache[ir.l]
        else:
            D = wigner_D(ir.l, alpha, beta, gamma)
        D_list.append(D)
        starts.append(int(sl.start))
        mults.append(int(mult))
        dims.append(int(ir.dim))
        apply_parity.append(ir.p == -1)

    return _rotate_irreps_core(x, D_list, parity, starts, mults, dims, apply_parity)


# ── internal helpers ────────────────────────────────────────────────────


_log_fact_cache: Dict[Tuple[int, torch.dtype, torch.device], Tensor] = {}


@torch.no_grad()
def _log_factorial_table(n: int, dtype=torch.float32, device=None) -> Tensor:
    """Return a tensor t where t[k] = log(k!) for k = 0 … n.  Cached.

    Default dtype is ``torch.float32`` — the project's runtime precision.
    For the tiny tables used here (size ≤ 4·l+1 ≤ 25 for l ≤ 6) fp32
    accumulates ``log(k!)`` with < 1 ULP error.  Pass ``dtype=torch.float64``
    explicitly when strict validation demands it."""
    device = device or torch.device("cpu")
    key = (n, dtype, device)
    if key not in _log_fact_cache:
        t = torch.zeros(n + 1, dtype=dtype, device=device)
        for k in range(1, n + 1):
            t[k] = t[k - 1] + math.log(k)
        _log_fact_cache[key] = t
    return _log_fact_cache[key]


def _pow_sign(base_sign: Tensor, exponent: Tensor) -> Tensor:
    """Sign of base^exponent when base is ±1 or 0.

    Returns +1 if base >= 0 or exponent is even, -1 if base < 0 and
    exponent is odd (integer exponents assumed).
    """
    neg = base_sign < 0
    odd = (exponent.long() % 2) != 0
    return torch.where(neg & odd, -torch.ones_like(base_sign), torch.ones_like(base_sign))


_cob_cache: Dict[Tuple[int, torch.dtype, torch.device], Tensor] = {}


def _change_of_basis_real_complex(
    l: int, dtype=torch.float32, device=None
) -> Tensor:
    """Unitary matrix U that converts complex spherical harmonics to real.

    Y^l_real = U · Y^l_complex

    Uses the Condon-Shortley convention::

        m > 0:  Y^real_m  = (1/√2) [(-1)^m Y^c_m + Y^c_{-m}]
        m = 0:  Y^real_0  = Y^c_0
        m < 0:  Y^real_m  = (i/√2) [(-1)^{|m|+1} Y^c_{|m|} + Y^c_{-|m|}]

    Parameters
    ----------
    l : int
        Degree.

    Returns
    -------
    Tensor (complex)
        Shape ``(2l+1, 2l+1)`` — unitary.  Cached per (l, dtype, device).
    """
    device = device or torch.device("cpu")
    key = (l, dtype, device)
    if key in _cob_cache:
        return _cob_cache[key]

    cdtype = torch.complex128 if dtype in (torch.float64, torch.complex128) else torch.complex64
    dim = 2 * l + 1
    U = torch.zeros(dim, dim, dtype=cdtype, device=device)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    for m in range(-l, l + 1):
        row = m + l  # row index in the (2l+1) block
        if m > 0:
            # Y_real_m  = (1/√2)[ (-1)^m Y_complex_m  +  Y_complex_{-m} ]
            U[row, m + l] = inv_sqrt2 * ((-1) ** m)
            U[row, -m + l] = inv_sqrt2
        elif m == 0:
            U[row, l] = 1.0
        else:  # m < 0
            # Y_real_m = (i/√2)[ (-1)^{|m|+1} Y_complex_{|m|} + Y_complex_{-|m|} ]
            am = abs(m)
            U[row, am + l] = 1j * inv_sqrt2 * ((-1) ** (am + 1))
            U[row, -am + l] = 1j * inv_sqrt2

    _cob_cache[key] = U
    return U


def _det3x3(R: Tensor) -> Tensor:
    """Explicit elementwise determinant of a batch of 3x3 matrices.

    Replaces ``torch.det`` (``aten._linalg_det``, an LU-based op whose AOTAutograd
    joint backward extracts a scalar via ``local_scalar_dense`` -> an unbacked
    symbolic float that breaks dynamic-shape ``torch.compile`` of the force-matching
    2nd backward). Pure arithmetic here has a clean, dynamic-safe backward and is
    bit-identical for 3x3.  Shape ``(*batch, 3, 3)`` -> ``(*batch,)``.
    """
    return (
        R[..., 0, 0] * (R[..., 1, 1] * R[..., 2, 2] - R[..., 1, 2] * R[..., 2, 1])
        - R[..., 0, 1] * (R[..., 1, 0] * R[..., 2, 2] - R[..., 1, 2] * R[..., 2, 0])
        + R[..., 0, 2] * (R[..., 1, 0] * R[..., 2, 1] - R[..., 1, 1] * R[..., 2, 0])
    )


def _batched_block_diag(blocks: List[Tensor], batch: torch.Size) -> Tensor:
    """Build a batch of block-diagonal matrices from a list of blocks.

    Each block has shape ``(*batch, n_i, n_i)``; the output has shape
    ``(*batch, N, N)`` where ``N = Σ n_i``.
    """
    N = sum(b.shape[-1] for b in blocks)
    out = torch.zeros(*batch, N, N, dtype=blocks[0].dtype, device=blocks[0].device)
    offset = 0
    for b in blocks:
        n = b.shape[-1]
        out[..., offset : offset + n, offset : offset + n] = b
        offset += n
    return out
