"""
CSG-SKin / DST-Skin path resolution, shared by paper-3 scripts.

paper-3 runs in two layouts: as a sibling of CSG-SKin on a development
machine, and nested inside it on the GPU server. Hardcoding
Path(__file__).resolve().parents[2] / "CSG-SKin" assumes the first and
raises ModuleNotFoundError under the second.

Each repo root is located by marker file instead, either as an ancestor of
the calling script or as a repo-named sibling of one, which covers both
layouts without either script knowing which it is in. CSG_SKIN_ROOT and
DST_SKIN_ROOT override the search when set.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

_CSG_MARKER = Path("src") / "utils" / "ood_metrics.py"
_DST_MARKER = Path("src") / "utils" / "ood_vim_react.py"


def _find_repo_root(script_file: Union[str, Path], repo_name: str, marker: Path, env_var: str) -> Path:
    env_override = os.environ.get(env_var)
    if env_override:
        candidate = Path(env_override).resolve()
        if (candidate / marker).is_file():
            return candidate
        raise RuntimeError(f"{env_var}={env_override!r} does not contain {marker} -- check the path.")

    start = Path(script_file).resolve().parent
    for ancestor in [start, *start.parents]:
        if (ancestor / marker).is_file():
            return ancestor
        sibling = ancestor.parent / repo_name
        if sibling != ancestor and (sibling / marker).is_file():
            return sibling

    raise RuntimeError(
        f"Could not locate {repo_name}'s repository root starting from {start} "
        f"(looked for it as an ancestor directory and as a {repo_name!r}-named "
        f"sibling of every ancestor). Set the {env_var} environment variable "
        "explicitly if this deployment uses a different layout."
    )


def find_csg_skin_root(script_file: Union[str, Path]) -> Path:
    return _find_repo_root(script_file, "CSG-SKin", _CSG_MARKER, "CSG_SKIN_ROOT")


def find_dst_skin_root(script_file: Union[str, Path]) -> Path:
    return _find_repo_root(script_file, "DST-Skin", _DST_MARKER, "DST_SKIN_ROOT")
