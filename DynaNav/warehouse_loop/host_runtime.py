"""Resolve and probe the logged-in host desktop before starting Isaac Python."""

import ctypes
import os
from pathlib import Path
import sys


def desktop_candidates(environ, proc=Path("/proc"), uid=None):
    uid = os.getuid() if uid is None else uid
    runtime = f"/run/user/{uid}"
    override = environ.get("TICVLA_DISPLAY")
    authority = environ.get("TICVLA_XAUTHORITY")
    if override:
        # An explicit session selection must not silently fall back elsewhere.
        return [
            (
                override,
                authority or environ.get("XAUTHORITY", runtime + "/gdm/Xauthority"),
                runtime,
            )
        ]
    candidates = []
    for entry in sorted(proc.iterdir()):
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            if (entry / "comm").read_text().strip() not in {
                "gnome-shell",
                "plasmashell",
                "xfce4-session",
                "mate-session",
                "cinnamon",
            }:
                continue
            env = dict(
                item.split("=", 1)
                for item in (entry / "environ").read_bytes().decode().split("\0")
                if "=" in item
            )
            if env.get("DISPLAY"):
                candidates.append(
                    (
                        env["DISPLAY"],
                        authority or env.get("XAUTHORITY", runtime + "/gdm/Xauthority"),
                        env.get("XDG_RUNTIME_DIR", runtime),
                    )
                )
        except (OSError, UnicodeError, ValueError):
            continue
    if environ.get("DISPLAY"):
        candidates.append(
            (
                environ["DISPLAY"],
                authority or environ.get("XAUTHORITY", runtime + "/gdm/Xauthority"),
                runtime,
            )
        )
    return list(dict.fromkeys(candidates))


def connect_display(display, authority):
    os.environ["XAUTHORITY"] = authority
    x11 = ctypes.CDLL("libX11.so.6")
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    connection = x11.XOpenDisplay(display.encode())
    if not connection:
        return False
    x11.XCloseDisplay(connection)
    return True


def main():
    launcher, script, *args = sys.argv[1:]
    if "--gui" in args or "--check" in args:
        for display, authority, runtime in desktop_candidates(os.environ):
            if connect_display(display, authority):
                os.environ.update(
                    DISPLAY=display, XAUTHORITY=authority, XDG_RUNTIME_DIR=runtime
                )
                print(
                    f"Host desktop connection OK: DISPLAY={display}, XAUTHORITY={authority}",
                    flush=True,
                )
                break
        else:
            raise SystemExit(
                "Cannot connect to the host desktop. Set TICVLA_DISPLAY and TICVLA_XAUTHORITY for the logged-in session; Isaac was not started."
            )
    if args and args[0] == "--check":
        os.execv(
            launcher,
            [
                launcher,
                "-c",
                "import os,platform,sys; print('Python:',sys.executable); print('libc:',platform.libc_ver()); print('uid:',os.getuid()); print('repository:',os.getcwd()); print('DISPLAY:',os.environ.get('DISPLAY'))",
            ],
        )
    os.execv(launcher, [launcher, script, *args])


if __name__ == "__main__":
    main()
