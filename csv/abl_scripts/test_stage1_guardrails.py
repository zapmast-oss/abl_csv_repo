"""Targeted Stage 1 tests; these tests do not run any production pipeline."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import unittest
from collections import Counter
from datetime import date
from pathlib import Path

import abl_path_policy
import z_abl_batch_manifest_build as batch_manifest
import z_abl_current_state_preflight_validate as preflight
import z_abl_source_registry_build as source_registry


REPO_ROOT = Path(__file__).resolve().parents[2]
OOTP_ROOT = REPO_ROOT / "csv" / "ootp_csv"
STAGE0_MANIFEST = (
    REPO_ROOT / "docs" / "repo_cleanup_stage0_ootp_sha256_2026_08_14.csv"
)
STAGE0_COMMIT = "a8ceb23b87e9f270621fbd36e35c57deabc5e20f"


def historical_blob(relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{STAGE0_COMMIT}:{relative}"], cwd=REPO_ROOT)


def historical_csv(relative: str) -> list[dict[str, str]]:
    data = historical_blob(relative)
    if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
        oid = data.decode("ascii").split("oid sha256:")[1].splitlines()[0]
        common = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"], cwd=REPO_ROOT, text=True
        ).strip()
        data = (REPO_ROOT / common / "lfs/objects" / oid[:2] / oid[2:4] / oid).read_bytes()
        if hashlib.sha256(data).hexdigest() != oid:
            raise ValueError(f"Invalid historical LFS payload: {relative}")
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class AuthoritativeInputTests(unittest.TestCase):
    def test_stage0_manifest_still_matches_exactly_72_immediate_csvs(self) -> None:
        expected_rows = read_csv(STAGE0_MANIFEST)
        expected = {row["filename"]: row for row in expected_rows}
        actual_paths = sorted(OOTP_ROOT.glob("*.csv"), key=lambda path: path.name.lower())

        self.assertEqual(72, len(expected_rows))
        self.assertEqual(72, len(actual_paths))
        self.assertEqual(set(expected), {path.name for path in actual_paths})
        # The August 14 byte receipt authenticates its original export. The
        # immediate 72-file contract continues to apply to today's export above.
        for name, entry in expected.items():
            obj = historical_blob(f"csv/ootp_csv/{name}")
            if obj.startswith(b"version https://git-lfs.github.com/spec/v1"):
                _, hash_line, size_line = obj.decode("ascii").splitlines()
                self.assertEqual(hash_line, "oid sha256:" + entry["sha256"], name)
                self.assertEqual(size_line, "size " + entry["size_bytes"], name)
            else:
                self.assertEqual(hashlib.sha256(obj).hexdigest(), entry["sha256"], name)
                self.assertEqual(len(obj), int(entry["size_bytes"]), name)


class AuthorityBoundaryTests(unittest.TestCase):
    def test_authoritative_predicate_is_immediate_only(self) -> None:
        self.assertTrue(
            abl_path_policy.is_authoritative_ootp_csv("csv/ootp_csv/games.csv")
        )
        self.assertFalse(
            abl_path_policy.is_authoritative_ootp_csv(
                "csv/ootp_csv/out/csv_out/z_ABL_Manager_Tendencies.csv"
            )
        )

    def test_rebuilt_registry_logic_has_72_authoritative_ootp_sources(self) -> None:
        catalog = read_csv(REPO_ROOT / "csv" / "out" / "docs" / "abl_data_catalog.csv")
        rows = [source_registry.make_row(row["file_path"], row) for row in catalog]
        counts = Counter(row["source_family"] for row in rows)
        self.assertEqual(72, counts["ootp_csv"])

        nested = next(
            row
            for row in rows
            if row["file_path"]
            == "csv/ootp_csv/out/csv_out/z_ABL_Manager_Tendencies.csv"
        )
        self.assertEqual("generated_pollution", nested["source_family"])
        self.assertEqual("derived_output", nested["authority_level"])
        self.assertEqual("no", nested["can_drive_current_state"])
        self.assertEqual("no", nested["can_support_current_state"])

    def test_batch_manifest_corrects_stale_registry_boundary(self) -> None:
        checked_in = historical_csv("csv/out/control/abl_source_registry.csv")
        self.assertEqual(
            73, sum(row["source_family"] == "ootp_csv" for row in checked_in)
        )
        governed = batch_manifest.select_governed_sources(checked_in)
        counts = Counter(row["source_family"] for row in governed)
        self.assertEqual(72, counts["ootp_csv"])
        self.assertEqual(20, counts["sortable_stats"])
        self.assertNotIn(
            "csv/ootp_csv/out/csv_out/z_ABL_Manager_Tendencies.csv",
            {row["file_path"] for row in governed},
        )
        current = read_csv(REPO_ROOT / "csv/out/control/abl_source_registry.csv")
        self.assertEqual(
            {row["file_path"] for row in governed},
            {row["file_path"] for row in batch_manifest.select_governed_sources(current)},
        )


class OutputPathPolicyTests(unittest.TestCase):
    def test_protected_and_accidental_destinations_are_rejected(self) -> None:
        root = REPO_ROOT / ".stage1_path_policy_test"
        rejected = (
            "csv/ootp_csv/out/new.csv",
            "csv/abl_statistics/generated.csv",
            "csv/statsplus/current/generated.csv",
            "csv/in/almanac_core/generated.csv",
            "csv/csv/out/generated.csv",
            "out/generated.csv",
        )
        for relative in rejected:
            with self.subTest(relative=relative):
                with self.assertRaises(abl_path_policy.PathPolicyError):
                    abl_path_policy.validate_output_path(relative, root)

    def test_canonical_csv_out_is_allowed(self) -> None:
        root = REPO_ROOT / ".stage1_path_policy_test"
        expected = (root / "csv" / "out" / "control" / "report.json").resolve()
        self.assertEqual(
            expected,
            abl_path_policy.validate_output_path(
                "csv/out/control/report.json", root
            ),
        )


class AsOfCutoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # July 24 was a partial day in this fixture. Subsequent exports
        # legitimately include additional completed July 24 games.
        cls.games = historical_csv("csv/ootp_csv/games.csv")

    def test_july_23_excludes_july_24_and_july_24_can_include_it(self) -> None:
        july23 = date(1981, 7, 23)
        july24 = date(1981, 7, 24)
        through23 = preflight.select_completed_games(self.games, 1981, july23, "200")
        through24 = preflight.select_completed_games(self.games, 1981, july24, "200")

        self.assertEqual(1163, len(through23))
        self.assertEqual(1165, len(through24))
        self.assertEqual(july23, max(preflight.parse_game_date(row["date"]) for row in through23))
        self.assertEqual(july24, max(preflight.parse_game_date(row["date"]) for row in through24))
        self.assertTrue(
            all(preflight.parse_game_date(row["date"]) <= july23 for row in through23)
        )
        self.assertGreater({row["game_id"] for row in through24}, {row["game_id"] for row in through23})

        batch23 = batch_manifest.select_completed_games(self.games, 1981, july23, "200")
        self.assertEqual(
            {row["game_id"] for row in through23},
            {row["game_id"] for row in batch23},
        )

    def test_cutoff_labels_match_requested_dates(self) -> None:
        output_dir = Path("control")
        self.assertEqual(
            "current_state_preflight_1981_target_1981-07-23",
            preflight.output_stem(output_dir, 1981, date(1981, 7, 23)).name,
        )
        self.assertEqual(
            "abl_batch_manifest_1981_asof_1981-07-24",
            batch_manifest.output_stem(output_dir, 1981, date(1981, 7, 24)).name,
        )


class ConfigurationSafetyTests(unittest.TestCase):
    def test_vscode_has_no_unsafe_default_runner(self) -> None:
        tasks_text = (REPO_ROOT / ".vscode" / "tasks.json").read_text(
            encoding="utf-8-sig"
        )
        tasks = json.loads(tasks_text)
        self.assertNotIn("_tmp_run_all.py", tasks_text)
        self.assertFalse(
            any(task.get("group", {}).get("isDefault") for task in tasks.get("tasks", []))
        )

        keybindings = (REPO_ROOT / ".vscode" / "keybindings.json").read_text(
            encoding="utf-8-sig"
        )
        self.assertNotIn("Run ABL Prep + Matchup", keybindings)


if __name__ == "__main__":
    unittest.main(verbosity=2)
