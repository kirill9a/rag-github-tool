import os
import shutil
from git import Repo

CLONE_DIR = "./cloned_repos"
IGNORE_DIRS = {'.git', 'node_modules', 'venv', '__pycache__', 'dist', 'build'}


def clone_repository(repo_url: str) -> str:
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    target_path = os.path.join(CLONE_DIR, repo_name)

    if os.path.exists(target_path):
        print(f"Removing old version of {repo_name}...")
        shutil.rmtree(target_path)

    print(f"Cloning {repo_url}...")
    Repo.clone_from(repo_url, target_path)
    print("Done.")

    return target_path


def build_project_tree(repo_path: str) -> list[str]:
    file_list = []

    for root, dirs, files in os.walk(repo_path):
        # skip noise
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, repo_path)
            file_list.append(relative_path)

    return file_list


if __name__ == "__main__":
    TEST_URL = "https://github.com/JuliusBrussee/caveman"

    path = clone_repository(TEST_URL)
    files = build_project_tree(path)

    print(f"\nFound {len(files)} files")
    print("First 10:")
    for f in files[:10]:
        print(f"  {f}")