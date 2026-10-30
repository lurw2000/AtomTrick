"""Radial smearing functions, adapted from the Open Catalyst Project (OCP).

Adapted (condensed, not a verbatim copy) from ``ocpmodels/models/scn/smearing.py`` of
https://github.com/Open-Catalyst-Project/ocp (now FAIR-Chem/fairchem), which EquiformerV2
imports.  Original licence notice (LICENSE.md as of tag v0.1.0):

    MIT License

    Copyright (c) Facebook, Inc. and its affiliates.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""
import math
import torch
import torch.nn as nn


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, basis_width_scalar=1.0):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)
        self.num_output = num_gaussians

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class SigmoidSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_sigmoid=50, basis_width_scalar=1.0):
        super().__init__()
        offset = torch.linspace(start, stop, num_sigmoid)
        self.coeff = (basis_width_scalar / (offset[1] - offset[0])).item()
        self.register_buffer("offset", offset)
        self.num_output = num_sigmoid

    def forward(self, dist):
        exp = self.coeff * (dist.view(-1, 1) - self.offset.view(1, -1))
        return torch.sigmoid(exp)


class LinearSigmoidSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_sigmoid=50, basis_width_scalar=1.0):
        super().__init__()
        offset = torch.linspace(start, stop, num_sigmoid)
        self.coeff = (basis_width_scalar / (offset[1] - offset[0])).item()
        self.register_buffer("offset", offset)
        self.num_output = num_sigmoid

    def forward(self, dist):
        exp = self.coeff * (dist.view(-1, 1) - self.offset.view(1, -1))
        return 0.5 * (1 + exp / (1 + exp.abs()))


class SiLUSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_output=50, basis_width_scalar=1.0):
        super().__init__()
        self.linear = nn.Linear(1, num_output)
        self.act = nn.SiLU()
        self.num_output = num_output

    def forward(self, dist):
        return self.act(self.linear(dist.view(-1, 1)))
