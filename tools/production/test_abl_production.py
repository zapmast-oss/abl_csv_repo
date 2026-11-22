"""Exercise checkpoint and release recovery without running production operations."""

from dataclasses import replace
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from abl_production_workflow import CSV_STAGES, Paths, Session, read_json


class ProductionSessionTests(unittest.TestCase):
    def setUp(self):
        local = Paths.local()
        scratch = (local.repo / "tmp").resolve()
        scratch.mkdir(exist_ok=True)
        self.folder = scratch / ("abl-command-center-tests-" + uuid4().hex)
        self.folder.mkdir()
        self.addCleanup(self.clean_fixture, scratch)
        self.paths = replace(
            local, live_export=self.folder / "live.lg/import_export/csv",
            backup=self.folder / "backup.lg", control=self.folder / "control",
            snapshots=self.folder / "snapshots", newsroom=self.folder / "newsroom",
        )
        self.paths.live_export.mkdir(parents=True)
        for name in ("world.dat", "teams.dat", "players.dat", "flag_save_completed.dat"):
            (self.paths.save / name).write_text(name, encoding="utf-8")

    def clean_fixture(self, scratch):
        if self.folder.resolve().parent == scratch and self.folder.name.startswith("abl-command-center-tests-"):
            shutil.rmtree(self.folder)

    def session(self, mode="quick"):
        return Session.create(self.paths, mode, "post-day", "1981-08-20", "August 19 final slate")

    def native_copy(self, _script, *args):
        self.paths.backup.mkdir(exist_ok=True)
        log = args[args.index("-LogFile") + 1]
        log.write_text("Robocopy finished with exit 3.", encoding="utf-8")
        for file in self.paths.save.glob("*.dat"):
            (self.paths.backup / file.name).write_bytes(file.read_bytes())
        return 3

    def test_quick_play_closes_with_verified_backup_and_deferrals(self):
        session = self.session()
        with patch("abl_production_workflow.ask", return_value=True), patch(
            "abl_production_workflow.powershell", side_effect=self.native_copy
        ) as backup:
            self.assertTrue(session.run())
            self.assertEqual(backup.call_count, 1)
        saved = read_json(session.file)
        checkpoint = read_json(session.folder / "checkpoint.json")
        self.assertTrue(saved["closed_at"])
        self.assertEqual(checkpoint["backup"]["verified_dat_files"], 4)
        self.assertEqual(checkpoint["save_hashes"], saved["save_hashes"])
        self.assertTrue(all(saved["stages"][stage]["status"] == "Deferred" for stage in CSV_STAGES))
        with patch("abl_production_workflow.ask", return_value=False), patch.object(session, "backup") as backup:
            self.assertTrue(session.run(resumed=True))
            backup.assert_not_called()
        self.assertEqual(read_json(session.folder / "checkpoint.json"), checkpoint)

    def test_backup_failure_blocks_capture_and_close(self):
        session = self.session("standard")
        with patch("abl_production_workflow.ask", return_value=True), patch(
            "abl_production_workflow.powershell", return_value=16
        ), patch.object(session, "capture_csv") as capture:
            self.assertFalse(session.run())
            capture.assert_not_called()
        self.assertEqual(session.state["stages"]["backup"]["status"], "Failed")
        self.assertIsNone(session.state["closed_at"])
        self.assertFalse((session.folder / "checkpoint.json").exists())

    def test_failed_validation_blocks_sync_and_desk_but_preserves_checkpoint(self):
        session = self.session("standard")
        with patch("abl_production_workflow.ask", return_value=True), patch(
            "abl_production_workflow.powershell", side_effect=self.native_copy
        ), patch("builtins.input", side_effect=["1981-08-19", ""]), patch.object(
            session, "capture_csv", return_value=("Captured fixture.", [])
        ), patch.object(session, "validate_csv", side_effect=ValueError("Mismatched finals")), patch.object(
            session, "sync_input"
        ) as sync, patch.object(session, "prepare_desk") as desk:
            self.assertTrue(session.run())
            sync.assert_not_called()
            desk.assert_not_called()
        self.assertEqual(session.state["stages"]["validate"]["status"], "Failed")
        self.assertEqual(session.state["stages"]["journalism"]["status"], "Deferred")
        self.assertTrue(session.succeeded("close"))

    def test_resumed_export_refuses_a_different_save(self):
        session = self.session()
        with patch("abl_production_workflow.ask", return_value=True), patch(
            "abl_production_workflow.powershell", side_effect=self.native_copy
        ):
            self.assertTrue(session.run())
        (self.paths.save / "world.dat").write_text("New world", encoding="utf-8")
        with patch("abl_production_workflow.ask", return_value=True), patch(
            "builtins.input", side_effect=["1981-08-19", ""]
        ):
            self.assertTrue(session.run(resumed=True))
        attempt = session.state["stages"]["capture"]
        self.assertEqual(attempt["status"], "Failed")
        self.assertIn("save has changed", attempt["message"])

    def test_original_attempt_is_recorded_when_running_stage_is_recovered(self):
        session = self.session()
        session.update("save", "Running", "Interrupted fixture")
        self.assertTrue(session.execute("save", session.record_save))
        attempts = session.state["stages"]["save"]["attempts"]
        self.assertEqual([attempt["status"] for attempt in attempts], ["Running", "Failed", "Running", "Succeeded"])

    def test_featured_media_failure_can_leave_a_closed_checkpoint(self):
        session = self.session("featured")
        with patch("abl_production_workflow.ask", side_effect=[True, False, True]), patch(
            "abl_production_workflow.powershell", side_effect=self.native_copy
        ), patch.object(session, "backup_video", side_effect=FileNotFoundError("H: unavailable")) as media:
            self.assertTrue(session.run())
            media.assert_called_once()
        self.assertEqual(session.state["stages"]["video"]["status"], "Failed")
        self.assertTrue(session.succeeded("close"))

    def test_release_requires_exact_approval_and_prevents_duplicate_receipts(self):
        session = self.session()
        session.update("backup", "Succeeded", "Fixture backup verified")
        artifact = self.folder / "master.mp4"
        artifact.write_bytes(b"reviewed master")
        key = session.queue_release("video", "SBV", "Series footage", artifact)
        self.assertEqual(session.state["stages"]["review"]["status"], "Requires approval")
        with self.assertRaisesRegex(ValueError, "Validate"):
            session.queue_release("editorial", "Forum", "Story", artifact)
        with self.assertRaisesRegex(ValueError, "already has a record"):
            session.queue_release("video", "sbv", "Duplicate master", artifact)
        with self.assertRaisesRegex(ValueError, "must have approval"):
            session.record_publication(key, "https://example.invalid/footage")
        session.approve_release(key, "Commissioner")
        artifact.write_bytes(b"changed master")
        with self.assertRaisesRegex(ValueError, "artifact has changed"):
            session.approved_artifact(key)
        artifact.write_bytes(b"reviewed master")
        self.assertEqual(session.approved_artifact(key)["approval"]["reviewer"], "Commissioner")
        session.record_publication(key, "https://example.invalid/footage")
        self.assertTrue(session.succeeded("publish"))
        with self.assertRaisesRegex(ValueError, "no existing publication receipt"):
            session.record_publication(key, "https://example.invalid/duplicate")
        artifact.write_bytes(b"new cut")
        new = session.queue_release("video", "SBV", "Second cut", artifact)
        session.approve_release(new, "Commissioner")
        with self.assertRaisesRegex(ValueError, "already recorded"):
            session.record_publication(new, "https://example.invalid/footage")


if __name__ == "__main__":
    unittest.main()
