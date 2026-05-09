"""Download SoccerNet-Tracking archives from the official Nextcloud host.

The ``SoccerNet`` pip package still calls legacy WebDAV URLs that return **401**
on current servers. These files are published as **password-protected public
shares**; we authenticate with a POST (requesttoken + password) then ``GET`` the
``download?files=…`` URL.

Tokens match ``SoccerNet/Downloader.py`` comments. Default share password is
``SoccerNet`` (same as ``downloadDataTask`` default), not ``s0cc3rn3t``.
"""

from __future__ import annotations

import re
from pathlib import Path

import requests

NC_HOST = "https://exrcsdrive.kaust.edu.sa"
# https://exrcsdrive.kaust.edu.sa/index.php/s/o9tzUs2GcuEwcnr
TRACKING_MAIN_TOKEN = "o9tzUs2GcuEwcnr"
# https://exrcsdrive.kaust.edu.sa/index.php/s/qWNjAzjEI6hezNf
TRACKING_LABEL_TOKEN = "qWNjAzjEI6hezNf"

_SPLIT_FILES: dict[str, tuple[str, str]] = {
    "train": (TRACKING_MAIN_TOKEN, "train.zip"),
    "test": (TRACKING_MAIN_TOKEN, "test.zip"),
    "challenge": (TRACKING_MAIN_TOKEN, "challenge.zip"),
    "test_labels": (TRACKING_LABEL_TOKEN, "test_labels.zip"),
    "challenge_labels": (TRACKING_LABEL_TOKEN, "challenge_labels.zip"),
}


def _parse_request_token(html: str) -> str:
    m = re.search(r'data-requesttoken="([^"]+)"', html)
    if m:
        return m.group(1)
    m = re.search(r'name="requesttoken"\s+value="([^"]+)"', html)
    if m:
        return m.group(1)
    raise ValueError("Could not parse Nextcloud requesttoken from authenticate page")


def _share_login(session: requests.Session, token: str, password: str) -> None:
    auth_url = f"{NC_HOST}/index.php/s/{token}/authenticate"
    r = session.get(auth_url, timeout=120)
    r.raise_for_status()
    rt = _parse_request_token(r.text)
    r = session.post(
        auth_url,
        data={"password": password, "requesttoken": rt},
        timeout=120,
        allow_redirects=True,
    )
    r.raise_for_status()
    if "/authenticate" in r.url and "Wrong password" in r.text:
        raise RuntimeError(
            f"Nextcloud share login failed for token {token[:4]}… "
            "Try password 'SoccerNet' (official default) or your NDA password."
        )


def _download_one(
    session: requests.Session,
    token: str,
    filename: str,
    dest: Path,
) -> None:
    if dest.is_file() and dest.stat().st_size > 10_000 and dest.read_bytes()[:2] == b"PK":
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{NC_HOST}/index.php/s/{token}/download?files={filename}&path=/"
    with session.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "application/zip" not in ct and "application/octet-stream" not in ct:
            raise RuntimeError(
                f"Expected a zip from {url}, got content-type={ct!r}. "
                "Wrong share password or file missing — try --password SoccerNet."
            )
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    if dest.read_bytes()[:2] != b"PK":
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is not a zip: {dest}")


def download_tracking_zips(data_dir: Path, splits: list[str], password: str) -> None:
    """Write ``<data_dir>/tracking/<split>.zip`` for each entry in ``splits``."""
    by_token: dict[str, list[str]] = {}
    for sp in splits:
        if sp not in _SPLIT_FILES:
            raise ValueError(f"Unknown tracking split {sp!r}; expected one of {list(_SPLIT_FILES)}")
        token, fname = _SPLIT_FILES[sp]
        by_token.setdefault(token, []).append(fname)

    tracking_dir = data_dir / "tracking"
    for token, fnames in by_token.items():
        session = requests.Session()
        _share_login(session, token, password)
        for fname in fnames:
            dest = tracking_dir / fname
            print(f"Downloading {fname} -> {dest} …", flush=True)
            _download_one(session, token, fname, dest)
