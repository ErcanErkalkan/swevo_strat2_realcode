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


# Development defaults remain backward compatible. Submission/finalization
# explicitly locks manuscript-facing artifacts with:
#   SWEVO_GENERATED_DIR=generated_submission
GENERATED = _project_path_from_env("SWEVO_GENERATED_DIR", ROOT / "generated")


def ensure_generated_dirs() -> None:
    # Only create the selected namespace. Historical checkpoint/archive/figure
    # subdirectories are not part of the current submission evidence surface.
    GENERATED.mkdir(parents=True, exist_ok=True)
