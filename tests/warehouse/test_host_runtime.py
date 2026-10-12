import os
from pathlib import Path
import tempfile
import unittest

from DynaNav.warehouse_loop.host_runtime import desktop_candidates


class DesktopSelectionTests(unittest.TestCase):
    def test_current_desktop_precedes_stale_display(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            session = proc / "123"
            session.mkdir()
            (session / "comm").write_text("gnome-shell\n")
            (session / "environ").write_bytes(
                b"DISPLAY=:1\0XAUTHORITY=/session/auth\0XDG_RUNTIME_DIR=/session\0"
            )
            result = desktop_candidates(
                {"DISPLAY": ":0", "XAUTHORITY": "/old/auth"}, proc
            )
            self.assertEqual(result[0], (":1", "/session/auth", "/session"))

    def test_explicit_session_never_falls_back(self):
        result = desktop_candidates(
            {"TICVLA_DISPLAY": ":99", "TICVLA_XAUTHORITY": "/explicit", "DISPLAY": ":1"}
        )
        self.assertEqual(result, [(":99", "/explicit", f"/run/user/{os.getuid()}")])

    def test_no_desktop_does_not_guess_display_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(desktop_candidates({}, Path(directory)), [])

    def test_other_users_desktops_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            session = proc / "123"
            session.mkdir()
            (session / "comm").write_text("gnome-shell\n")
            (session / "environ").write_bytes(b"DISPLAY=:1\0")
            self.assertEqual(desktop_candidates({}, proc, uid=os.getuid() + 1), [])


if __name__ == "__main__":
    unittest.main()
