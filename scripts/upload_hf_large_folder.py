#!/usr/bin/env python3
"""Resumable Hugging Face upload for large feature directories."""
import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("folder", type=Path)
    parser.add_argument("remote_path")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not args.folder.is_dir():
        raise SystemExit(f"missing folder: {args.folder}")
    # upload_large_folder keeps a local upload cache and resumes after an
    # interrupted network transfer; files already present on the Hub are
    # skipped by the Hub commit logic.
    HfApi().upload_large_folder(
        repo_id=args.repo_id,
        folder_path=args.folder,
        repo_type="dataset",
        num_workers=args.workers,
        print_report=True,
        print_report_every=60,
    )


if __name__ == "__main__":
    main()
