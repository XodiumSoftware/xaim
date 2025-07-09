from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import requests


class Utils:
    """Utility methods for various tasks."""

    @staticmethod
    def getRepo(url: str) -> str:
        """
        Clones a GitHub repository from a URL and extracts the contents.

        Args:
            url: GitHub repository URL (e.g. "https://github.com/user/repo").

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
                if file.is_file()
            )
        
    @staticmethod
    def getRepos(org: str) -> list[str]:
        """
        Fetches all public repository clone URLs for a given GitHub organization.

        Args:
            org: GitHub organization name.

        Returns:
            A list of HTTPS clone URLs (e.g. "https://github.com/org/repo.git").
        """
        return [
            repo["clone_url"]
            for page in range(1, 100) 
            if (res := requests.get(
                f"https://api.github.com/orgs/{org}/repos",
                params={"per_page": 100, "page": page},
                headers={"Accept": "application/vnd.github.v3+json"}
            )).ok and (repos := res.json())
            for repo in repos
        ]