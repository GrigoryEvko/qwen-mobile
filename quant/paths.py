"""The paths of the project tree that more than one module of the pipeline uses."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
# The llama.cpp tree: the submodule of this repository, then a plain clone beside it for a machine
# that has no submodule. The first tree that holds a path gives it.
LLAMA_DIRS = (ROOT / "third_party" / "llama.cpp", ROOT / "llama.cpp")


def llama_path(*parts: str) -> Path:
    """Give a path in the llama.cpp tree.

    Args:
        parts: The parts of the path in the tree, for example "gguf-py"

    Returns:
        The path in the first tree of LLAMA_DIRS that holds it. If no tree
        holds it, the path in the submodule, thus an error gives the
        expected location.
    """
    for tree in LLAMA_DIRS:
        path = tree.joinpath(*parts)
        if path.exists():
            return path
    return LLAMA_DIRS[0].joinpath(*parts)


def gguf_module(tree: Path | None = None) -> ModuleType:
    """Import the gguf package of a llama.cpp tree and give the module.

    The package is not on the path of the environment as a copy: pyproject.toml takes it from the
    submodule as an editable path dependency, and a caller that runs without that environment
    needs the directory on sys.path.

    The function puts the directory on sys.path one time for each process. An insertion for each
    call would make sys.path grow through a run that exports many files.

    Args:
        tree: The root of a llama.cpp tree, or None for the tree of LLAMA_DIRS

    Returns:
        The gguf module

    Raises:
        ImportError: If the directory holds no gguf package
    """
    path = str(tree / "gguf-py" if tree is not None else llama_path("gguf-py"))
    if path not in sys.path:
        sys.path.insert(0, path)
    import gguf  # noqa: PLC0415

    return gguf
