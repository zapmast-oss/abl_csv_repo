import pathlib
import subprocess

root = pathlib.Path(r'C:\Users\earld\OneDrive\Documents\Out of the Park Developments\OOTP Baseball 26\saved_games\Action Baseball League.lg\import_export\csv')
data_base = root / 'ootp_csv'
scripts = sorted((root / 'abl_scripts').glob('z_abl_*.py'), key=lambda p: p.name)
failures = []
for script in scripts:
    print(f'Running {script.name}...')
    proc = subprocess.run(['python', str(script), '--base', str(data_base)], cwd=root)
    if proc.returncode != 0:
        failures.append(script.name)
if failures:
    print('Scripts failed:', ', '.join(failures))
else:
    print('All z_abl scripts completed successfully.')
