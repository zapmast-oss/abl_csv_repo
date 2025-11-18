from pathlib import Path
import sys
import subprocess


def run_script(path: Path, label: str) -> None:
    if not path.exists():
        sys.stderr.write(f"[ERROR] {label} script not found: {path}\n")
        sys.exit(1)

    print(f"[INFO] Running {label}: {path}")
    result = subprocess.run([sys.executable, str(path)], text=True)
    if result.returncode != 0:
        sys.stderr.write(f"[ERROR] {label} failed with exit code {result.returncode}\n")
        sys.exit(result.returncode)
    print(f"[INFO] {label} completed successfully.\n")


def main():
    # repo root = two levels up from this file: .../abl_csv_repo
    root = Path(__file__).resolve().parents[2]

    scripts_dir = root / "csv" / "abl_scripts"
    out_text_dir = root / "csv" / "out" / "text_out"

    run_week_script = scripts_dir / "z_abl_run_week_1981.py"
    eb_pack_script = scripts_dir / "z_abl_eb_pack_1981_monday.py"

    # 1) Build Monday data
    run_script(run_week_script, "Weekly 1981 wrapper (z_abl_run_week_1981.py)")

    # 2) Build EB data pack
    run_script(eb_pack_script, "EB Monday pack (z_abl_eb_pack_1981_monday.py)")

    # 3) Point user to the EB text file
    eb_pack_path = out_text_dir / "eb_data_pack_1981_monday.txt"
    if not eb_pack_path.exists():
        sys.stderr.write(
            f"[ERROR] Expected EB Data Pack not found: {eb_pack_path}\n"
        )
        sys.exit(1)

    print("\n[INFO] TURNKEY COMPLETE.")
    print("[INFO] EB Data Pack is ready here:")
    print(f"       {eb_pack_path}\n")
    print("Next step:")
    print("  1) Open that file.")
    print("  2) Select all and copy.")
    print('  3) Paste into ChatGPT and say: "EB – here is the data pack for Monday. Write the column."')


if __name__ == "__main__":
    main()
