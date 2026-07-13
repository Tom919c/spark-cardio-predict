"""Upload a local dataset to an HDFS raw directory configured through .env."""

from __future__ import annotations

import argparse
from pathlib import Path

from config.settings import BaseConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("local_path", type=Path)
    parser.add_argument("--hdfs-dir", default=BaseConfig.HDFS_RAW_PATH)
    return parser.parse_args()


def upload_to_hdfs(local_path: Path, hdfs_dir: str):
    from hdfs import InsecureClient

    if not local_path.exists():
        raise FileNotFoundError(f"Local file does not exist: {local_path}")
    client = InsecureClient(url=BaseConfig.HDFS_WEB_URL, user=BaseConfig.HDFS_USER)
    client.makedirs(hdfs_dir)
    destination = f"{hdfs_dir.rstrip('/')}/{local_path.name}"
    client.upload(destination, str(local_path), overwrite=True)
    print(destination)


if __name__ == "__main__":
    arguments = parse_args()
    upload_to_hdfs(arguments.local_path, arguments.hdfs_dir)
