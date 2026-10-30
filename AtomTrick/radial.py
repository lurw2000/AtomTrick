"""Radial basis functions for distance expansion.

**ExpNormalSmearing** — Gaussian basis in exponential-distance space
multiplied by a cosine cutoff, with learnable centres and widths
(PhysNet [Unke & Meuwly, 2019] basis in the parameterisation of
TorchMD-Net [Thölke & De Fabritiis, 2022], also used by GotenNet).
This is the basis of the paper model (``rbf_type="expnorm"``).

The features are smooth, differentiable and go to zero at the cutoff
distance, so forces remain continuous.  In the attention layers the
basis feeds a bias-free MLP, so the radial gates vanish at the cutoff too.

Usage::

    rbf = ExpNormalSmearing(num_basis=16, cutoff=5.0)
    d = torch.tensor([1.0, 2.5, 4.9])
    features = rbf(d)  # (3, 16)
"""

# ExpNormalSmearing and CosineCutoff are adapted from TorchMD-Net
# (https://github.com/torchmd/torchmd-net, torchmdnet/models/utils.py),
# Copyright (c) 2021-2023 Universitat Pompeu Fabra, MIT License.
# See THIRD_PARTY_NOTICES.md for the full notice.

import math

import torch
import torch.nn as nn
from torch import Tensor


class CosineCutoff(nn.Module):
    r"""Smooth cosine cutoff envelope.

    .. math::

        c(r) = \frac{1}{2}\bigl(\cos(\pi r / r_{\text{cut}}) + 1\bigr)
               \cdot \mathbf{1}[r < r_{\text{cut}}]

    Parameters
    ----------
    cutoff : float
        Cutoff radius.
    """

    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, d: Tensor) -> Tensor:
        cutoffs = 0.5 * (torch.cos(d * math.pi / self.cutoff) + 1.0)
        return cutoffs * (d < self.cutoff).to(cutoffs.dtype)


class ExpNormalSmearing(nn.Module):
    r"""Exponential normal smearing radial basis (TorchMD-Net / GotenNet).

    Gaussian basis functions placed in exponential-distance space,
    multiplied by a cosine cutoff envelope:

    .. math::

        \phi_k(r) = c(r) \exp\bigl(-\beta_k
                    \bigl(e^{-\alpha r} - \mu_k\bigr)^2\bigr)

    where :math:`\alpha = 5 / r_{\text{cut}}`, the means :math:`\mu_k`
    are linearly spaced from :math:`e^{-r_{\text{cut}}}` to 1, and
    :math:`\beta_k = (2 / n \cdot (1 - e^{-r_{\text{cut}}}))^{-2}`.

    Parameters
    ----------
    num_basis : int
        Number of radial basis functions.
    cutoff : float
        Cutoff radius.
    learnable : bool
        If ``True``, means and betas are learnable parameters.
    """

    def __init__(
        self,
        num_basis: int = 16,
        cutoff: float = 5.0,
        learnable: bool = True,
    ):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if learnable:
            self.means = nn.Parameter(means)
            self.betas = nn.Parameter(betas)
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.num_basis)
        betas = torch.tensor(
            [(2 / self.num_basis * (1 - start_value)) ** -2] * self.num_basis
        )
        return means, betas

    def forward(self, d: Tensor) -> Tensor:
        """Expand distances into radial basis features.

        Parameters
        ----------
        d : Tensor, shape ``(...,)``
            Inter-atomic distances (positive scalars).

        Returns
        -------
        Tensor, shape ``(..., num_basis)``
        """
        d_unsq = d.unsqueeze(-1)                               # (..., 1)
        cutoff_val = self.cutoff_fn(d_unsq)                    # (..., 1)
        exp_term = torch.exp(self.alpha * (-d_unsq))           # (..., 1)
        return cutoff_val * torch.exp(
            -self.betas * (exp_term - self.means) ** 2
        )                                                      # (..., B)

    def extra_repr(self) -> str:
        return (
            f"num_basis={self.num_basis}, "
            f"cutoff={self.cutoff}, "
            f"learnable={isinstance(self.means, nn.Parameter)}"
        )
