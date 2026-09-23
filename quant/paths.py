"""The paths of the project tree that more than one module of the pipeline uses."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The llama.cpp tree: the submodule of this repository, then the plain clone that the GPU box keeps in
# ~/qwen-mobile/llama.cpp. The first tree that holds a path gives it.
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
