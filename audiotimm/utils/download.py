from __future__ import annotations

import sys
import urllib.request
from pathlib import Path
from typing import Optional


def get_cache_dir() -> Path:
    """Return (and create) the audiotimm weight cache directory."""
    d = Path.home() / ".cache" / "audiotimm"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_file(url: str, dest: Path, desc: str = "") -> None:
    """Download *url* to *dest*, skipping if dest already exists.

    Shows a simple inline progress indicator using only stdlib.
    Cleans up the partial file if the download fails.
    """
    if dest.exists():
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    label = desc or url.split("/")[-1]
    print(f"Downloading {label} ...", flush=True)

    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=_make_hook(label))
    except Exception as exc:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    # newline after the inline progress
    print(flush=True)


def _make_hook(label: str):
    def hook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / 1_048_576
            total_mb = total_size / 1_048_576
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            sys.stdout.write(
                f"\r  [{bar}] {pct:3d}%  {mb:.1f}/{total_mb:.1f} MB  {label}"
            )
            sys.stdout.flush()

    return hook
