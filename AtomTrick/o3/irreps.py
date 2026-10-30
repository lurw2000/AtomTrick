from __future__ import annotations

import re
from typing import Dict, List, Tuple, Union, Iterator

import torch
from torch import Tensor

from .utils import _build_ch_info


class Irrep:
    """A single O(3) irreducible representation of degree l and parity p.

    For degree l there are 2l+1 components: m = -l, -l+1, ..., l.
    Parity p is +1 (even, 'e') or -1 (odd, 'o').

    String notation: ``"0e"``, ``"1o"``, ``"2e"``, etc.

    All derived attributes (dim, parity_str) are stored as plain slot
    attributes set at construction — no per-call recomputation.
    """

    __slots__ = ("l", "p", "dim", "parity_str")

    def __init__(self, l: int, p: int):
        if not isinstance(l, int) or l < 0:
            raise ValueError(f"l must be a non-negative integer, got {l!r}")
        if p not in (1, -1):
            raise ValueError(f"p must be +1 or -1, got {p!r}")
        self.l = l
        self.p = p
        self.dim = 2 * l + 1
        self.parity_str = "e" if p == 1 else "o"

    @classmethod
    def from_string(cls, s: str) -> "Irrep":
        """Parse a string like ``'0e'``, ``'1o'``, ``'2e'``."""
        m = re.fullmatch(r"(\d+)([eo])", s.strip())
        if not m:
            raise ValueError(f"Cannot parse Irrep from string: {s!r}")
        return cls(int(m.group(1)), 1 if m.group(2) == "e" else -1)

    def __repr__(self) -> str:
        return f"Irrep(l={self.l}, p={self.p})"

    def __str__(self) -> str:
        return f"{self.l}{self.parity_str}"

    def __eq__(self, other) -> bool:
        if isinstance(other, Irrep):
            return self.l == other.l and self.p == other.p
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.l, self.p))

    def __lt__(self, other: "Irrep") -> bool:
        return (self.l, self.p) < (other.l, other.p)


class Irreps:
    """An ordered list of O(3) irreps with multiplicities.

    Each entry is a ``(multiplicity, Irrep)`` pair.  Total channel count::

        dim = sum(mult * (2l+1)  for mult, irrep in self)

    All derived data (dim, slices, lmax, ls, parities, index maps) are
    computed once at construction and stored as plain attributes — repeated
    access is a single attribute lookup.

    Construction
    ------------
    From a string::

        Irreps("1x0e + 3x1o + 2x2e")

    From ``(multiplicity, l, p)`` or ``(multiplicity, Irrep)`` tuples::

        Irreps([(1, 0, 1), (3, 1, -1), (2, 2, 1)])
        Irreps([(1, Irrep(0, 1)), (3, Irrep(1, -1))])

    Copy from another ``Irreps``::

        Irreps(other)
    """

    def __init__(self, irreps: Union[str, List, "Irreps"]):
        if isinstance(irreps, Irreps):
            self._entries: List[Tuple[int, Irrep]] = list(irreps._entries)
        elif isinstance(irreps, str):
            self._entries = self._parse(irreps)
        elif isinstance(irreps, (list, tuple)):
            self._entries = []
            for item in irreps:
                if isinstance(item, (list, tuple)):
                    if len(item) == 2:
                        mult, second = item
                        if isinstance(second, Irrep):
                            self._entries.append((int(mult), second))
                        else:
                            raise ValueError(
                                f"Expected (multiplicity, Irrep), got {item!r}"
                            )
                    elif len(item) == 3:
                        mult, l, p = item
                        self._entries.append((int(mult), Irrep(int(l), int(p))))
                    else:
                        raise ValueError(
                            f"Expected (mult, Irrep) or (mult, l, p), got {item!r}"
                        )
                else:
                    raise ValueError(f"Expected tuple/list entry, got {item!r}")
        else:
            raise TypeError(f"Cannot construct Irreps from {type(irreps).__name__!r}")

        self._finalize()

    @staticmethod
    def _parse(s: str) -> List[Tuple[int, Irrep]]:
        entries = []
        for token in s.replace(" ", "").split("+"):
            if not token:
                continue
            m = re.fullmatch(r"(\d+)x(\d+)([eo])", token)
            if m:
                mult = int(m.group(1))
                l = int(m.group(2))
                p = 1 if m.group(3) == "e" else -1
            else:
                m = re.fullmatch(r"(\d+)([eo])", token)
                if m:
                    mult = 1
                    l = int(m.group(1))
                    p = 1 if m.group(2) == "e" else -1
                else:
                    raise ValueError(f"Cannot parse irrep token: {token!r}")
            entries.append((mult, Irrep(l, p)))
        return entries

    def _finalize(self):
        """Precompute all derived data once after _entries is set."""
        entries = self._entries

        # Single pass: slices, dim, ls, parities, index maps
        slices: List[slice] = []
        ls: List[int] = []
        parities: List[int] = []
        lp_idxs: Dict[Tuple[int, int], List[int]] = {}
        offset = 0
        for i, (mult, irrep) in enumerate(entries):
            width = mult * irrep.dim
            slices.append(slice(offset, offset + width))
            offset += width
            ls.append(irrep.l)
            parities.append(irrep.p)
            lp_key = (irrep.l, irrep.p)
            if lp_key in lp_idxs:
                lp_idxs[lp_key].append(i)
            else:
                lp_idxs[lp_key] = [i]

        self._slices: List[slice] = slices
        self.dim: int = offset
        self.ls: List[int] = ls
        self.parities: List[int] = parities
        self.lmax: int = max(ls) if ls else 0

        # Precompute channel access info per (l,p).
        # Each value is either a slice (contiguous → zero-copy view) or a
        # list[int] (non-contiguous → one gather, no cat loop).
        self._lp_ch_info: Dict[Tuple[int, int], object] = {
            key: _build_ch_info(idxs, slices) for key, idxs in lp_idxs.items()
        }

        # Cache hash and repr
        self._hash: int = hash(tuple((mult, ir.l, ir.p) for mult, ir in entries))
        self._repr: str = (
            " + ".join(f"{mult}x{irrep}" for mult, irrep in entries)
            if entries else "Irreps()"
        )

    # ------------------------------------------------------------------
    # Container interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Tuple[int, Irrep]]:
        return iter(self._entries)

    def __getitem__(self, idx: int) -> Tuple[int, Irrep]:
        return self._entries[idx]

    def __eq__(self, other) -> bool:
        if isinstance(other, Irreps):
            if self._hash != other._hash:   # fast path: different hash → not equal
                return False
            return self._entries == other._entries
        return NotImplemented

    def __hash__(self) -> int:
        return self._hash

    def __add__(self, other: "Irreps") -> "Irreps":
        """Concatenate two Irreps (direct bypass of the parsing loop)."""
        result = object.__new__(Irreps)
        result._entries = self._entries + other._entries
        result._finalize()
        return result

    def __repr__(self) -> str:
        return self._repr

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def slices(self) -> List[slice]:
        """Per-entry slices into the flat channel dimension (precomputed)."""
        return self._slices

    def simplified(self) -> "Irreps":
        """Merge entries that share the same (l, p), preserving first-seen order."""
        merged: Dict[Tuple[int, int], int] = {}
        order: List[Irrep] = []
        for mult, irrep in self._entries:
            key = (irrep.l, irrep.p)
            if key not in merged:
                order.append(irrep)
                merged[key] = mult
            else:
                merged[key] += mult
        result = object.__new__(Irreps)
        result._entries = [(merged[(ir.l, ir.p)], ir) for ir in order]
        result._finalize()
        return result

    def sort(self) -> "Irreps":
        """Return a new Irreps sorted by ascending l, then parity (e before o)."""
        result = object.__new__(Irreps)
        result._entries = sorted(self._entries, key=lambda x: (x[1].l, x[1].p))
        result._finalize()
        return result


