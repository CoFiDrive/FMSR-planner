"""Keep the legacy top-level Garage imports inside this frozen snapshot."""
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
GARAGE_ROOT = PACKAGE_ROOT / 'vendor/garage'


def configure_imports():
    for name in ('config', 'data', 'model', 'transfuser_utils', 'transfuser'):
        module = sys.modules.get(name)
        if module is not None and getattr(module, '__file__', None):
            if not Path(module.__file__).resolve().is_relative_to(GARAGE_ROOT):
                raise RuntimeError(f'{name} already comes from another checkout; run in a fresh process')
    paths = [GARAGE_ROOT, PACKAGE_ROOT, WORKSPACE_ROOT,
             WORKSPACE_ROOT / '3rd_party/CARLA_0915/PythonAPI/carla',
             WORKSPACE_ROOT / '3rd_party/Bench2Drive/leaderboard',
             WORKSPACE_ROOT / '3rd_party/Bench2Drive/scenario_runner']
    for path in reversed(paths):
        value = str(path)
        # Leaderboard inserts the agent's directory at index zero. Existing
        # PYTHONPATH entries must be moved ahead of it, not merely left in place.
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


configure_imports()
