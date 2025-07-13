
#%%
from pathlib import Path

def list_all_files(root_path):
    root = Path(root_path)
    excluded_exts = {".csv", ".json"}

    for path in root.rglob("*"):
        if (
            path.is_file()
            and not any(part.lower() in {"venv", ".venv"} for part in path.parts)
            and path.suffix.lower() not in excluded_exts
        ):
            print(path)

def list_all_directories_excluding_venv(root_path):
    root = Path(root_path)

    for path in root.rglob("*"):
        if (
            path.is_dir()
            and not any(part.lower() in {"venv", ".venv"} for part in path.parts)
        ):
            print(path)

# Example usage
list_all_files("C:/Users/kwasi/OneDrive/Documents/Personal Projects/schwab_trader")

list_all_directories_excluding_venv(
    "C:/Users/kwasi/OneDrive/Documents/Personal Projects/schwab_trader"
)
# %%
