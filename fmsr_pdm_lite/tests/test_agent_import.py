"""Exercise the actual leaderboard top-level module loading convention."""
import os
from pathlib import Path
import subprocess
import sys


def test_leaderboard_agent_directory_does_not_shadow_garage_model():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join([str(root), str(root/'vendor/garage')])
    environment['OMP_NUM_THREADS'] = '1'
    environment['OPENBLAS_NUM_THREADS'] = '1'
    code = '''
import importlib
import sys
from pathlib import Path
from fmsrpdm import bootstrap
# Simulate leaderboard, including a previously imported bootstrap module.
sys.path.insert(0, str(bootstrap.PACKAGE_ROOT/'fmsrpdm'))
agent = importlib.import_module('agent')
import model
import config
assert agent.get_entry_point() == 'FMSRPDMAgent'
assert Path(model.__file__).resolve() == bootstrap.GARAGE_ROOT/'model.py'
assert Path(config.__file__).resolve() == bootstrap.GARAGE_ROOT/'config.py'
assert sys.path[0] == str(bootstrap.GARAGE_ROOT)
print('leaderboard agent import passed')
'''
    result = subprocess.run([sys.executable, '-c', code], env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'leaderboard agent import passed' in result.stdout
