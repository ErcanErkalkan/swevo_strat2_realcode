from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
TEMPLATES = ROOT / "templates"
DOCS = ROOT / "docs"


def _project_path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else ROOT / path


# Generated evidence has one default namespace. SWEVO_GENERATED_DIR
# may point a reproduction copy to another output directory.
GENERATED = _project_path_from_env("SWEVO_GENERATED_DIR", ROOT / "results" / "primary")


def ensure_generated_dirs() -> None:
    # Only create the selected namespace. Historical checkpoint/archive/figure
    # subdirectories are not part of the current submission evidence surface.
    GENERATED.mkdir(parents=True, exist_ok=True)
