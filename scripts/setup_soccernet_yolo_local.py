#!/usr/bin/env python3
"""Download SoccerNet-Tracking locally, unzip archives, export YOLO player dataset.

Uses the **SoccerNet** pip package only — same pattern as ``notebooks/colab_atharv.ipynb``::

    from SoccerNet.Downloader import SoccerNetDownloader
    dl = SoccerNetDownloader(LocalDirectory=DATA_DIR)
    dl.downloadDataTask(task=\"tracking\", split=[\"train\", \"test\", \"challenge\"], password=PASSWORD)

The helper only **downloads** ``tracking/*.zip``; this script **extracts** them to
``<data-dir>/tracking/<split>/SNMOT-*/…``, then runs ``export_yolo``.

From **repository root** (after ``pip install SoccerNet`` and ``pip install -e .``)::

    python scripts/setup_soccernet_yolo_local.py \\
        --data-dir ~/data/soccernet \\
        --export-out ~/data/yolo_player_export

Default ``--password`` matches the notebook's ``PASSWORD = \"s0cc3rn3t\"``. If you get HTTP 401,
try the other Colab variant: ``--password SoccerNet`` (library default, first commented line there).
Add ``test_labels`` / ``challenge_labels`` to ``--download-splits`` if you need those zips too.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _seq_dirs(split_dir: Path) -> list[Path]:
    if not split_dir.is_dir():
        return []
    return [p for p in split_dir.iterdir() if p.is_dir() and (p / "seqinfo.ini").is_file()]


def _split_ready(tracking_root: Path, split: str) -> bool:
    return len(_seq_dirs(tracking_root / split)) > 0


def _move_loose_snmot_into_split(tracking_root: Path, split: str) -> None:
    """If an archive dumped ``SNMOT-*`` folders at ``tracking/`` root, move under ``split/``."""
    split_dir = tracking_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for p in list(tracking_root.iterdir()):
        if not p.is_dir() or not p.name.startswith("SNMOT-"):
            continue
        target = split_dir / p.name
        if target.exists():
            continue
        p.rename(target)


def _extract_zip(archive: Path, tracking_root: Path, split_for_loose: str | None) -> None:
    print(f"Extracting {archive.name} into {tracking_root} ...", flush=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(tracking_root)
    if split_for_loose is not None:
        _move_loose_snmot_into_split(tracking_root, split_for_loose)


def extract_tracking_archives(
    tracking_root: Path,
    *,
    splits: list[str],
    force: bool,
) -> None:
    """Unpack ``train.zip`` / ``test.zip`` / ``challenge.zip`` (and label packs) under ``tracking/``."""
    tracking_root.mkdir(parents=True, exist_ok=True)
    main_splits = [s for s in splits if s in ("train", "test", "challenge")]

    for sp in main_splits:
        z = tracking_root / f"{sp}.zip"
        if not z.is_file():
            continue
        if not force and _split_ready(tracking_root, sp):
            print(
                f"Skip extract {z.name}: {tracking_root / sp!s} already has sequences.", flush=True
            )
            continue
        _extract_zip(z, tracking_root, sp)

    for name in ("test_labels", "challenge_labels"):
        z = tracking_root / f"{name}.zip"
        if not z.is_file():
            continue
        print(f"Extracting label pack {z.name} ...", flush=True)
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(tracking_root)


def _download_tracking(data_dir: Path, splits: list[str], password: str) -> None:
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError as e:
        raise SystemExit(
            "Missing SoccerNet. Install with: pip install SoccerNet\n"
            "(only needed for the download step; YOLO export uses this repo.)"
        ) from e

    data_dir.mkdir(parents=True, exist_ok=True)
    dl = SoccerNetDownloader(LocalDirectory=str(data_dir))
    dl.downloadDataTask(task="tracking", split=splits, password=password)


def _run_export(
    *,
    tracking_root: Path,
    export_split: str,
    export_out: Path,
    val_fraction: float,
    frame_stride: int,
    max_seqs: int | None,
    copy_images: bool,
    no_progress: bool,
    force_recopy: bool,
    no_prefer_gameinfo_ball: bool,
    seed: int,
) -> None:
    repo = _repo_root()
    cmd: list[str] = [
        sys.executable,
        "-m",
        "soccernet_detect.export_yolo",
        "--tracking-root",
        str(tracking_root),
        "--split",
        export_split,
        "--out",
        str(export_out),
        "--val-fraction",
        str(val_fraction),
        "--frame-stride",
        str(frame_stride),
        "--seed",
        str(seed),
    ]
    if copy_images:
        cmd.append("--copy-images")
    if no_progress:
        cmd.append("--no-progress")
    if force_recopy:
        cmd.append("--force-recopy")
    if no_prefer_gameinfo_ball:
        cmd.append("--no-prefer-gameinfo-ball")
    if max_seqs is not None:
        cmd.extend(["--max-seqs", str(max_seqs)])

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo, check=True)


def main(argv: list[str] | None = None) -> None:
    default_data = Path(os.environ.get("SOCCERNET_DIR", "~/data/soccernet")).expanduser()
    default_out = Path(os.environ.get("YOLO_EXPORT_DIR", "~/data/yolo_player_export")).expanduser()
    # Match colab_atharv.ipynb commented downloadDataTask(split=["train", "test", "challenge"])
    default_dl_splits = ["train", "test", "challenge"]

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=default_data,
        help="SoccerNet root: zips land in <dir>/tracking/ (default: ~/data/soccernet)",
    )
    p.add_argument(
        "--download-splits",
        nargs="+",
        default=default_dl_splits,
        metavar="SPLIT",
        help="Forwarded to SoccerNet downloadDataTask(task='tracking', split=...) (default: train test challenge)",
    )
    p.add_argument("--skip-download", action="store_true", help="Do not call SoccerNet downloader")
    p.add_argument(
        "--skip-extract",
        action="store_true",
        help="Do not unzip tracking/*.zip (use if you already have SNMOT-* folders)",
    )
    p.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract split zips even if sequences look present",
    )
    p.add_argument(
        "--skip-export",
        action="store_true",
        help="Download/extract only; do not run YOLO export",
    )
    p.add_argument(
        "--password",
        default=os.environ.get("SOCCERNET_PASSWORD", "s0cc3rn3t"),
        help="downloadDataTask password=… (default: s0cc3rn3t like colab_atharv PASSWORD; override env SOCCERNET_PASSWORD)",
    )
    p.add_argument(
        "--export-split",
        default="train",
        help="Which tracking folder to export (train|test|challenge). Default: train",
    )
    p.add_argument("--export-out", type=Path, default=default_out, help="YOLO dataset output root")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-seqs", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--symlink-images",
        action="store_true",
        help="Symlink JPEGs in YOLO layout instead of copying",
    )
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--force-recopy", action="store_true")
    p.add_argument(
        "--no-prefer-gameinfo-ball",
        action="store_true",
        help="Forward to export_yolo",
    )
    args = p.parse_args(argv)

    data_dir = args.data_dir.expanduser().resolve()
    tracking_root = data_dir / "tracking"
    export_out = args.export_out.expanduser().resolve()

    if not args.skip_download:
        print(
            f"Downloading tracking splits {args.download_splits!r} into {data_dir} ...", flush=True
        )
        _download_tracking(data_dir, list(args.download_splits), str(args.password))
    elif not tracking_root.is_dir():
        raise SystemExit(f"Missing {tracking_root}. Drop --skip-download or fix --data-dir.")

    if not args.skip_extract:
        # Unpack *.zip even when --skip-download (e.g. user downloaded zips elsewhere).
        main_for_extract = [s for s in args.download_splits if s in ("train", "test", "challenge")]
        if args.skip_download and not main_for_extract:
            main_for_extract = ["train", "test", "challenge"]
        extract_tracking_archives(
            tracking_root,
            splits=main_for_extract,
            force=args.force_extract,
        )

    export_split = args.export_split
    split_dir = tracking_root / export_split
    if not args.skip_export:
        if not _split_ready(tracking_root, export_split):
            raise SystemExit(
                f"No sequences with seqinfo.ini under {split_dir}. "
                f"Finish download/extract or set --export-split. "
                f"If zips are present, run without --skip-extract."
            )
        copy_images = not args.symlink_images
        _run_export(
            tracking_root=tracking_root,
            export_split=export_split,
            export_out=export_out,
            val_fraction=args.val_fraction,
            frame_stride=args.frame_stride,
            max_seqs=args.max_seqs,
            copy_images=copy_images,
            no_progress=args.no_progress,
            force_recopy=args.force_recopy,
            no_prefer_gameinfo_ball=args.no_prefer_gameinfo_ball,
            seed=args.seed,
        )
        print(
            f"\nDone. YOLO root: {export_out}\n"
            f"Train: yolo detect train data={export_out / 'data.yaml'} …",
            flush=True,
        )
    else:
        print(f"Skipping YOLO export. Tracking tree: {tracking_root}", flush=True)


if __name__ == "__main__":
    main()
