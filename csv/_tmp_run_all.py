import os
import pathlib
import subprocess
from subprocess import DEVNULL, STDOUT

DEFAULT_DATA_ROOT = pathlib.Path(
    r'C:\Users\earld\OneDrive\Documents\Out of the Park Developments\OOTP Baseball 26\saved_games\Action Baseball League.lg\import_export\csv'
)
script_dir = pathlib.Path(__file__).resolve().parent
configured = os.environ.get('ABL_CSV_EXEC_ROOT')
data_root_candidate = pathlib.Path(configured) if configured else DEFAULT_DATA_ROOT
if (data_root_candidate / 'ootp_csv').exists():
    data_base = (data_root_candidate / 'ootp_csv').resolve()
elif data_root_candidate.exists():
    data_base = data_root_candidate.resolve()
else:
    data_base = (script_dir / 'ootp_csv').resolve()
scripts = sorted((script_dir / 'abl_scripts').glob('z_abl_*.py'), key=lambda p: p.name)
failures = []
for script in scripts:
    print(f'Running {script.name}...')
    proc = subprocess.run(
        ['python', str(script), '--base', str(data_base)],
        cwd=script_dir,
        stdout=DEVNULL,
        stderr=STDOUT,
    )
    if proc.returncode != 0:
        failures.append(script.name)
if failures:
    print('Scripts failed:', ', '.join(failures))
else:
    print('All z_abl scripts completed successfully.')
