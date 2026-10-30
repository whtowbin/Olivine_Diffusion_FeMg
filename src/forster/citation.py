"""How to cite FORSTER: ``forster.cite()``, ``forster cite`` and the app's Help menu.

The same facts are in CITATION.cff at the repository root (tests/test_citation.py checks
they agree). The reference for a diffusion equation is the equation's own ``source`` text
in `forster.diffusivity`, so there is one copy of it.
"""

from __future__ import annotations

from importlib.metadata import version

from forster.diffusivity import Diffusivity, resolve

TITLE = "FORSTER: Fitting Olivine Residence & Storage Timescales from Elemental Relaxation"
# TODO(Henry): add co-authors (Jollands, Shi): full names, ORCIDs, affiliations; and decide
# on Plank and the order
AUTHORS = ("Towbin, W. Henry",)
URL = "https://forster.readthedocs.io"
REPOSITORY = "https://github.com/whtowbin/Olivine_Diffusion_FeMg"
# TODO(Henry): after the first GitHub release Zenodo mints a DOI; set it here and in
# CITATION.cff
DOI: str | None = None
PAPER = (
    "Towbin, W. H. and Plank, T. (2026). Decompression Melting of Peridotite Xenoliths. "
    "Journal of Petrology 67(9), egag077. doi:10.1093/petrology/egag077"
)
PAPER_DOI = "10.1093/petrology/egag077"


def cite(diffusivity: Diffusivity | str | None = None) -> str:
    """Plain text on how to cite FORSTER, the paper, and (when ``diffusivity`` is a name,
    an alias such as ``"dipra"`` or an equation) the source of that diffusion equation."""
    doi = f"doi:{DOI}. " if DOI else ""
    software = (
        f"{'; '.join(AUTHORS)}. {TITLE} (version {version('forster')}) [software]. "
        f"{doi}{REPOSITORY}. Documentation: {URL}"
    )
    parts = [
        f"To cite the software:\n  {software}",
        f"To cite the method:\n  {PAPER}",
    ]
    if diffusivity is not None:
        equation = resolve(diffusivity)
        source = getattr(equation, "source", "") or "no source recorded for this equation"
        parts.append(f"To cite the diffusion equation you used ({equation.name}):\n  {source}")
    return "\n\n".join(parts) + "\n"
