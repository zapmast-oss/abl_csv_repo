"""Targeted Stage 1 tests; these tests do not run any production pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        for path in actual_paths:
            self.assertEqual(int(expected[path.name]["size_bytes"]), path.stat().st_size)
            self.assertEqual(expected[path.name]["sha256"], sha256(path))


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
        checked_in = read_csv(
            REPO_ROOT / "csv" / "out" / "control" / "abl_source_registry.csv"
        )
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
        cls.games = read_csv(OOTP_ROOT / "games.csv")

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
