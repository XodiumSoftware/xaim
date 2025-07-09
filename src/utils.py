from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


class Utils:
  """Utility methods for various tasks."""

  @staticmethod
  def getRepo(url: str, exts: tuple[str, ...] | None = None) -> str:
    """
    Clones a GitHub repository from a URL and extracts the contents of files matching the given extensions.

    Args:
        url: GitHub repository URL (e.g. "https://github.com/user/repo").
        exts: File extensions to include (e.g. ".py", ".md"). If None, includes all files.

    Returns:
        Concatenated text content of all matched files.
    """
    with TemporaryDirectory() as tmpdir:
        repo_name = url.rstrip("/").split("/")[-1]
        repo_path = Path(tmpdir) / repo_name

        subprocess.run(["git", "clone", "--depth=1", url, str(repo_path)], check=True, stdout=subprocess.DEVNULL)

        return "\n".join(
            file.read_text(encoding="utf-8", errors="ignore")
            for file in repo_path.rglob("*")
            if file.is_file() and (exts is None or file.suffix in exts)
        )