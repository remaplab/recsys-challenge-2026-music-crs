from __future__ import annotations

import argparse
import logging
from pathlib import Path

from huggingface_hub import HfFileSystem, hf_hub_download


DATASET_FILES: dict[str, tuple[str, ...]] = {
	"talkpl-ai/TalkPlayData-Challenge-User-Metadata": (
		"data/all_users-00000-of-00001.parquet",
	),
	"talkpl-ai/TalkPlayData-Challenge-Track-Metadata": (
		"data/all_tracks-00000-of-00001.parquet",
		"data/test_tracks-00000-of-00001.parquet",
	),
	"talkpl-ai/TalkPlayData-Challenge-User-Embeddings": (
		"data/train-00000-of-00001.parquet",
		"data/test_warm-00000-of-00001.parquet",
		"data/test_cold-00000-of-00001.parquet",
	),
	"talkpl-ai/TalkPlayData-Challenge-Track-Embeddings": (
		"data/all_tracks-*.parquet",
		"data/test_tracks-00000-of-00001.parquet",
	),
	"talkpl-ai/TalkPlayData-Challenge-Dataset": (
		"data/train-00000-of-00001.parquet",
		"data/test-00000-of-00001.parquet",
	),
	"talkpl-ai/TalkPlayData-Challenge-Blind-A": (
		"data/test-00000-of-00001.parquet",
	),
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Download TalkPlay datasets from Hugging Face")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(__file__).resolve().parents[3] / "data",
		help="Local output directory (default: project_root/data)",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Re-download files even if they already exist locally",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Show which files would be downloaded without downloading",
	)
	return parser.parse_args()


def expand_patterns(repo_id: str, patterns: tuple[str, ...], fs: HfFileSystem) -> list[str]:
	files: list[str] = []
	for pattern in patterns:
		if "*" in pattern or "?" in pattern or "[" in pattern:
			remote_pattern = f"datasets/{repo_id}/{pattern}"
			matches = fs.glob(remote_pattern)
			prefix = f"datasets/{repo_id}/"
			files.extend(path.removeprefix(prefix) for path in sorted(matches))
		else:
			files.append(pattern)
	return files


def download_repo_files(
	repo_id: str,
	files: list[str],
	output_root: Path,
	overwrite: bool,
	dry_run: bool,
) -> tuple[int, int]:
	downloaded = 0
	skipped = 0
	repo_output_dir = output_root / repo_id

	for relative_file in files:
		destination = repo_output_dir / relative_file
		if destination.exists() and destination.stat().st_size > 0 and not overwrite:
			logging.info("SKIP %s", destination)
			skipped += 1
			continue

		destination.parent.mkdir(parents=True, exist_ok=True)
		if dry_run:
			logging.info("DRY-RUN download %s -> %s", f"{repo_id}/{relative_file}", destination)
			continue

		logging.info("DOWNLOAD %s", f"{repo_id}/{relative_file}")
		hf_hub_download(
			repo_id=repo_id,
			repo_type="dataset",
			filename=relative_file,
			local_dir=repo_output_dir,
			force_download=overwrite,
		)
		downloaded += 1

	return downloaded, skipped


def main() -> None:
	args = parse_args()
	logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
	logging.getLogger("httpx").setLevel(logging.WARNING)

	fs = HfFileSystem()
	total_downloaded = 0
	total_skipped = 0

	for repo_id, patterns in DATASET_FILES.items():
		files = expand_patterns(repo_id, patterns, fs)
		if not files:
			logging.warning("No files matched for repository %s", repo_id)
			continue

		downloaded, skipped = download_repo_files(
			repo_id=repo_id,
			files=files,
			output_root=args.output_dir,
			overwrite=args.overwrite,
			dry_run=args.dry_run,
		)
		total_downloaded += downloaded
		total_skipped += skipped

	logging.info(
		"Completed. Downloaded=%s, Skipped=%s, Output=%s",
		total_downloaded,
		total_skipped,
		args.output_dir,
	)


if __name__ == "__main__":
	main()
