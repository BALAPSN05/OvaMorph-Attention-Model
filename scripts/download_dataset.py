"""Download the selected Kaggle ovarian ultrasound dataset locally.

Requires Kaggle API authentication (KAGGLE_USERNAME/KAGGLE_KEY or kaggle.json).
The dataset is intentionally not committed to GitHub.
"""
from pathlib import Path
import argparse
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="ucimachinelearning/ovarian-ultrasound-image-dataset",
        help="Kaggle dataset slug",
    )
    parser.add_argument("--output_dir", default="data/ovarian_ultrasound")
    args = parser.parse_args()

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install kaggle") from exc

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {args.dataset} to {out} ...")
    api.dataset_download_files(args.dataset, path=str(out), unzip=True)
    print("Download complete. Inspect the extracted folder structure before training.")


if __name__ == "__main__":
    main()
