"""
Shared CSG-SKin path resolution for paper-3 scripts.

paper-3/ has already been deployed in two different layouts:
  - sibling: Research/{CSG-SKin, paper-3}/   (local development machine)
  - nested:  CSG-SKin/paper-3/               (GPU server deployment)

The original scripts hardcoded `Path(__file__).resolve().parents[2] / "CSG-SKin"`,
which assumes the sibling layout and breaks under the nested one
(ModuleNotFoundError: No module named 'src') -- this is exactly what happened
on first real server deployment. Neither script should hardcode a parent-count
at all; this module locates CSG-SKin's root by looking for a marker file
(src/utils/ood_metrics.py) either as an ancestor of the calling script, or as
a "CSG-SKin"-named sibling of any ancestor -- both layouts satisfy one of
these without assuming which.

An explicit CSG_SKIN_ROOT environment variable always wins, for any future
layout neither heuristic anticipates.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

_MARKER = Path("src") / "utils" / "ood_metrics.py"


def find_csg_skin_root(script_file: Union[str, Path]) -> Path:
    env_override = os.environ.get("CSG_SKIN_ROOT")
    if env_override:
        candidate = Path(env_override).resolve()
        if (candidate / _MARKER).is_file():
            return candidate
        raise RuntimeError(
            f"CSG_SKIN_ROOT={env_override!r} does not contain {_MARKER} -- check the path."
        )

    start = Path(script_file).resolve().parent
    for ancestor in [start, *start.parents]:
        if (ancestor / _MARKER).is_file():
            return ancestor
        sibling = ancestor.parent / "CSG-SKin"
        if sibling != ancestor and (sibling / _MARKER).is_file():
            return sibling

    raise RuntimeError(
        f"Could not locate CSG-SKin's repository root starting from {start} "
        "(looked for it as an ancestor directory and as a 'CSG-SKin'-named "
        "sibling of every ancestor). Set the CSG_SKIN_ROOT environment "
        "variable explicitly if this deployment uses a different layout."
    )
