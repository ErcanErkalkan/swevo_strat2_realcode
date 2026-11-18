from pathlib import Path
import hashlib,json,sys,platform,importlib.metadata
ROOT=Path(__file__).resolve().parents[1]


def metadata():
    files=[*sorted((ROOT/'src').rglob('*.py')),*sorted((ROOT/'protocol').glob('*.md'))]
    return {'source_hashes':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
        'environment_sha256':hashlib.sha256(json.dumps(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())).encode()).hexdigest(),
        'version':'C17.0_PARTIAL','telemetry_schema':'c17-events-v1'}
