from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OUT_DIR = SCRIPT_DIR / "out"
OUT_TXT = OUT_DIR / "text_out"
OUT_CSV = OUT_DIR / "csv_out"
OUT_TXT.mkdir(parents=True, exist_ok=True)
OUT_CSV.mkdir(parents=True, exist_ok=True)
print("OUT_TXT:", OUT_TXT)
print("OUT_CSV:", OUT_CSV)

import os
import subprocess
from subprocess import DEVNULL, STDOUT

DEFAULT_DATA_ROOT = Path(
    r"C:\Users\earld\OneDrive\Documents\Out of the Park Developments\OOTP Baseball 26\saved_games\Action Baseball League.lg\import_export\csv"
)


def resolve_data_root() -> Path:
    configured = os.environ.get("ABL_CSV_EXEC_ROOT")
    if configured:
        candidate = Path(configured)
        if candidate.exists():
            return candidate
    if DEFAULT_DATA_ROOT.exists():
        return DEFAULT_DATA_ROOT
    return SCRIPT_DIR / "ootp_csv"


def build_base_arg(data_root: Path) -> Path:
    if (data_root / "ootp_csv").exists():
        return (data_root / "ootp_csv").resolve()
    return data_root.resolve()


def main() -> None:
    data_root = resolve_data_root()
    data_base = build_base_arg(data_root)
    scripts = sorted((SCRIPT_DIR / "abl_scripts").glob("z_abl_*.py"), key=lambda p: p.name)
    failures: list[str] = []
    print(f"DATA_ROOT: {data_root}")
    print(f"--base passed to scripts: {data_base}")
    _start_cwd = os.getcwd()
    for script in scripts:
        print(f"Running {script.name}...")
        proc = subprocess.run(
            ["python", str(script), "--base", str(data_base)],
            cwd=SCRIPT_DIR,
            stdout=DEVNULL,
            stderr=STDOUT,
        )
        assert os.getcwd() == _start_cwd, "A report script changed CWD; remove os.chdir() in that script."
        if proc.returncode != 0:
            failures.append(script.name)
    print("Running manager parser...")
    parser_proc = subprocess.run(
        [
            "python",
            "csv/abl_scripts/parse_managers.py",
            "--index",
            "data_raw/ootp_html/history/league_200_all_managers_index.html",
        ],
        cwd=ROOT,
        stdout=DEVNULL,
        stderr=STDOUT,
    )
    if parser_proc.returncode != 0:
        failures.append("parse_managers.py")

    print("Validating managers...")
    validator_proc = subprocess.run(
        [
            "python",
            "csv/abl_scripts/validate_managers.py",
            "--src",
            "csv/out/csv_out/abl_managers_summary.csv",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if validator_proc.stdout:
        print(validator_proc.stdout.strip())
    if validator_proc.stderr:
        print(validator_proc.stderr.strip())
    if validator_proc.returncode != 0:
        failures.append("validate_managers.py")

    print("Generating manager one-pager...")
    onepager_proc = subprocess.run(
        [
            "python",
            "csv/abl_scripts/report_managers_onepager.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if onepager_proc.stdout:
        print(onepager_proc.stdout.strip())
    if onepager_proc.stderr:
        print(onepager_proc.stderr.strip())
    if onepager_proc.returncode != 0:
        failures.append("report_managers_onepager.py")

    prep_db = ROOT / "data_work" / "abl.db"
    if prep_db.exists():
        print("Generating broadcast prep packets...")
        prep_proc = subprocess.run(
            [
                "python",
                "csv/abl_scripts/report_broadcast_prep.py",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if prep_proc.stdout:
            print(prep_proc.stdout.strip())
        if prep_proc.stderr:
            print(prep_proc.stderr.strip())
        if prep_proc.returncode != 0:
            failures.append("report_broadcast_prep.py")
    else:
        print("Skipping broadcast prep (database missing)")

    matchup_home = os.environ.get("MATCHUP_HOME")
    matchup_away = os.environ.get("MATCHUP_AWAY")
    if matchup_home and matchup_away:
        print(f"Generating manager matchup card: {matchup_home} vs {matchup_away} (env)")
        matchup_proc = subprocess.run(["python", "csv/abl_scripts/report_manager_matchup.py", "--home", matchup_home, "--away", matchup_away], cwd=ROOT, capture_output=True, text=True)
        if matchup_proc.stdout:
            print(matchup_proc.stdout.strip())
        if matchup_proc.stderr:
            print(matchup_proc.stderr.strip())
        if matchup_proc.returncode != 0:
            failures.append("report_manager_matchup.py")
    elif matchup_home or matchup_away:
        print("MATCHUP_HOME and MATCHUP_AWAY must both be set; skipping matchup card.")
    else:
        print("No matchup env found; skipping manager matchup card.")
    if failures:
        print("Scripts failed:", ", ".join(failures))
    else:
        print("All z_abl scripts completed successfully.")


if __name__ == "__main__":
    main()




