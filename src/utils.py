from pathlib import Path


class Utils:
  """Utility methods for various tasks."""

  @staticmethod
  def getRepo(path: str, exts: tuple[str, ...] | None = None) -> str:
    """
    Recursively reads and concatenates contents of files matching given extensions.

    Args:
        path: Root directory of the repo.
        exts: File extensions to include (e.g. ".py", ".md"). If None, includes all files.

    Returns:
        All matched file contents as a single string.
    """
    return "\n".join(
            i.read_text(encoding="utf-8")
            for i in Path(path).rglob("*")
            if i.is_file() and (exts is None or i.suffix in exts)
        )