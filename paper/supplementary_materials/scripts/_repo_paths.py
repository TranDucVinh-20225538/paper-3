"""
Locate the audited classifier's repository, shared by the scripts/ pipeline.

The working version of this file resolves the repository by its directory
name as a fallback. That name is withheld for double-blind review, because
the repository is public under an account that identifies the authors, so
this version resolves it two ways only:

    1. the CLASSIFIER_ROOT environment variable (SECOND_CLASSIFIER_ROOT for
       the second repository), if set;
    2. otherwise, the nearest ancestor directory of the calling script that
       contains the repository's marker file.

Reviewers running stages 1-3 (see README) should set CLASSIFIER_ROOT
explicitly. Stages 4-9 never import this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

_CLASSIFIER_MARKER = Path("src") / "utils" / "ood_metrics.py"
_SECOND_CLASSIFIER_MARKER = Path("src") / "utils" / "ood_vim_react.py"


def _find(script_file: Union[str, Path], marker: Path, env_var: str) -> Path:
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

    raise RuntimeError(
        f"Could not locate the repository from {start}: no ancestor directory "
        f"contains {marker}. Set the {env_var} environment variable to its "
        "checkout. The repository itself is not included in this supplement "
        "and is withheld during review."
    )


def find_classifier_root(script_file: Union[str, Path]) -> Path:
    return _find(script_file, _CLASSIFIER_MARKER, "CLASSIFIER_ROOT")


def find_second_classifier_root(script_file: Union[str, Path]) -> Path:
    return _find(script_file, _SECOND_CLASSIFIER_MARKER, "SECOND_CLASSIFIER_ROOT")
