import json
import os
from pathlib import Path
import shutil
import sys
import unittest
import uuid
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from abl_newsroom_control import claim_operation, current_issue_matches, process


class ExportDetectionTests(unittest.TestCase):
    def test_changes_repeats_missing_files_retry_and_manual_run(self):
        scratch = Path(__file__).resolve().parents[2] / "tmp"
        workspace = scratch / ("export-control-check-" + uuid.uuid4().hex)
        workspace.mkdir(parents=True)
        try:
            source = workspace / "live"
            control = workspace / "control"
            source.mkdir()
            names = [f"input-{n:02d}.csv" for n in range(72)]
            for name in names:
                (source / name).write_bytes(b"test\r\n")
            run = Mock(return_value=0)
            available = Mock(return_value=True)
            def poll():
                return process("poll", source, names, control, run, available)
            receipt = control / "poll-state.json"
            self.assertEqual(poll(), 0)
            self.assertEqual(poll(), 0)
            self.assertEqual(run.call_count, 1)
            original = json.loads(receipt.read_text())["last_success_sha256"]
            file = source / names[0]
            timestamps = file.stat()
            file.write_bytes(b"edit\r\n")
            os.utime(file, ns=(timestamps.st_atime_ns,timestamps.st_mtime_ns))
            run.return_value = 1
            self.assertEqual(poll(), 1)
            self.assertEqual(json.loads(receipt.read_text())["last_success_sha256"],original)
            run.return_value = 0
            self.assertEqual(poll(), 0)
            changed = json.loads(receipt.read_text())["last_success_sha256"]
            self.assertNotEqual(changed,original)
            self.assertEqual(run.call_count, 3)
            file.unlink()
            self.assertEqual(poll(), 1)
            self.assertEqual(run.call_count, 3)
            file.write_bytes(b"edit\r\n")
            self.assertEqual(poll(), 0)
            self.assertEqual(run.call_count, 3)
            self.assertEqual(process("run",source,names,control,run,available),0)
            self.assertEqual(run.call_count, 4)
            with claim_operation(control) as acquired:
                self.assertTrue(acquired)
                self.assertEqual(poll(), 0)
                self.assertEqual(run.call_count, 4)
            available.return_value = False
            self.assertEqual(poll(), 1)
            self.assertEqual(run.call_count, 5)
            self.assertEqual(json.loads(receipt.read_text())["status"],"failed")
            self.assertFalse(current_issue_matches("mock",workspace))
        finally:
            if workspace.resolve().parent != scratch.resolve():
                raise RuntimeError("Unexpected test workspace")
            shutil.rmtree(workspace)


if __name__ == "__main__":
    unittest.main()
