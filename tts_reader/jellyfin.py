"""Copy finished videos to a Jellyfin server's library over a Windows (SMB)
share, then ask Jellyfin to rescan, so they can be cast from its phone app.

Set up once in ~/.config/bookcaster/jellyfin (mode 600):

    JELLYFIN_URL=http://<server>:8096
    JELLYFIN_KEY=<Dashboard -> API Keys>
    SMB_SHARE=//<server>/bookcaster     (the folder a Jellyfin library reads)
    SMB_USER=bookcaster
    SMB_PASS=...

Without that file nothing is sent. Each book gets a folder named after its
title; each video is named after its chapter ("08 The Adventure of the
Speckled Band.mp4"), and a remade video replaces the old one.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen

CONFIG = Path(os.environ.get("BOOKCASTER_JELLYFIN_CONFIG",
                             Path.home() / ".config" / "bookcaster" / "jellyfin")).expanduser()
# Capped so a big upload doesn't starve everything else on a Wi-Fi link (on
# the old Wi-Fi 5 link the Happier session timed out during a full-speed
# download, hence 8 MiB/s then). On Wi-Fi 7 (6 GHz, ~865 Mbit/s, 2026-10-16)
# 40 MiB/s leaves most of the link free.
RATE = int(float(os.environ.get("BOOKCASTER_JELLYFIN_MBPS", "40")) * 2**20)
CHUNK = 1 << 20


def config() -> dict | None:
    if not CONFIG.exists():
        return None
    cfg = dict(line.split("=", 1) for line in CONFIG.read_text().splitlines()
               if "=" in line and not line.lstrip().startswith("#"))
    cfg = {k.strip(): v.strip() for k, v in cfg.items()}
    return cfg if cfg.get("SMB_SHARE") and cfg.get("SMB_USER") else None


def _cred_file(cfg: dict) -> Path:
    """smbclient reads the login from a file, so the password never shows in ps."""
    p = CONFIG.with_name("smb.cred")
    want = f"username={cfg['SMB_USER']}\npassword={cfg.get('SMB_PASS', '')}\n"
    if not p.exists() or p.read_text() != want:
        p.touch(mode=0o600)
        p.chmod(0o600)
        p.write_text(want)
    return p


def _smb(cfg: dict, commands: str, stdin=None) -> subprocess.Popen:
    return subprocess.Popen(["smbclient", cfg["SMB_SHARE"], "-A", str(_cred_file(cfg)),
                             "-c", commands], stdin=stdin, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=stdin is None)


def safe_name(text: str) -> str:
    """A Windows-legal file or folder name."""
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text)
    return re.sub(r"\s+", " ", text).strip(" .")[:120] or "untitled"


def nice_title(text: str) -> str:
    """'VIII. THE ADVENTURE OF THE SPECKLED BAND' -> 'The Adventure of the Speckled Band'."""
    text = re.sub(r"^\s*(chapter\s+)?([ivxlc]+|\d+)[.:)]\s+", "", text.strip(), flags=re.I)
    if text.isupper():
        small = {"a", "an", "and", "as", "at", "but", "by", "for", "in", "of", "on", "or", "the", "to"}
        words = text.lower().split()
        text = " ".join(w if i and w in small else w[:1].upper() + w[1:] for i, w in enumerate(words))
    return text


def remote_size(cfg: dict, path: str) -> int | None:
    out = _smb(cfg, f'ls "{path}"').communicate()[0]
    name = path.rsplit("\\", 1)[-1]
    for line in out.splitlines():
        if line.strip().startswith(name):
            m = re.search(r"\s(\d+)\s+\w{3} \w{3}\s+\d+ [\d:]+ \d{4}$", line)
            if m:
                return int(m.group(1))
    return None


def send(src: Path, folder: str, name: str, log=print) -> bool:
    """Copy *src* to <share>/<folder>/<name>, rate-capped; skip if already there."""
    cfg = config()
    if cfg is None:
        return False
    folder, name = safe_name(folder), safe_name(name)
    dest = f"{folder}\\{name}"
    size = src.stat().st_size
    if remote_size(cfg, dest) == size:
        log(f"jellyfin: {dest} is already there")
        return True
    _smb(cfg, f'mkdir "{folder}"').communicate()
    tmp = f"{folder}\\.{name}.part"
    t0 = time.time()
    if remote_size(cfg, tmp) != size:     # an earlier upload may be waiting to be swapped in
        p = _smb(cfg, f'put /dev/stdin "{tmp}"', stdin=subprocess.PIPE)
        sent = 0
        with open(src, "rb") as f:
            while chunk := f.read(CHUNK):
                p.stdin.write(chunk)
                sent += len(chunk)
                ahead = sent / RATE - (time.time() - t0)
                if ahead > 0:
                    time.sleep(ahead)
        p.stdin.close()
        out = p.stdout.read().decode(errors="replace")
        if p.wait() != 0 or remote_size(cfg, tmp) != size:
            log(f"jellyfin: copying {name} failed: {out.strip()[-300:]}")
            return False
    # Windows won't replace a file that's open, and Jellyfin holds the old
    # copy while it's playing or being scanned: retry for a few minutes.
    for attempt in range(10):
        out = _smb(cfg, f'del "{dest}"; rename "{tmp}" "{dest}"').communicate()[0]
        if remote_size(cfg, dest) == size:
            log(f"jellyfin: sent {dest} ({size / 2**20:.0f} MiB in {time.time() - t0:.0f} s)")
            return True
        time.sleep(30)
    log(f"jellyfin: the old {name} is in use on the server (playing?); the new one is "
        f"uploaded and swaps in at the next `bookcaster jellyfin`: {out.strip()[-200:]}")
    return False


def _api(cfg: dict, method: str, path: str):
    req = Request(cfg["JELLYFIN_URL"].rstrip("/") + path, method=method,
                  headers={"Authorization": f'MediaBrowser Token="{cfg["JELLYFIN_KEY"]}"'})
    with urlopen(req, timeout=15) as r:
        body = r.read()
    return json.loads(body) if body else None


def rescan(log=print) -> None:
    """Ask Jellyfin to scan its libraries for new files. (Refreshing just the
    one library's item returned OK but never found the new video.)"""
    cfg = config()
    if not cfg or not cfg.get("JELLYFIN_URL") or not cfg.get("JELLYFIN_KEY"):
        return
    try:
        _api(cfg, "POST", "/Library/Refresh")
        log("jellyfin: library rescan started")
    except Exception as exc:
        log(f"jellyfin: rescan failed ({exc}); it will show up at the next scheduled scan")
