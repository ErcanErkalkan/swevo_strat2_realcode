from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
GENERATED = ROOT / "generated"
TEMPLATES = ROOT / "templates"
DOCS = ROOT / "docs"

def ensure_generated_dirs() -> None:
    for path in [
        GENERATED,
        GENERATED / "checkpoints",
        GENERATED / "archives",
        GENERATED / "figures",
    ]:
        path.mkdir(parents=True, exist_ok=True)
