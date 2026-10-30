"""Stage 1: download the four Mendeley Data NSE price archives.

Writes each dataset's zip to data/raw/<dataset_id>.zip, extracts its CSVs into
data/raw/<dataset_id>/, and writes one manifest.json recording the source URL, a sha256 of
each zip, and the row count of every extracted CSV, so a later stage can check it loaded
everything the download actually contained.

Without --refresh, an already-downloaded zip is reused as is, so a re-run does not hit
Mendeley again by accident.

    py -3 ingest/fetch_mendeley.py
    py -3 ingest/fetch_mendeley.py --refresh
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import requests

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"

# Four datasets from the same author (Barack Wanjawa, University of Nairobi), same CC BY 4.0
# licence, same 13-column schema, together spanning 2013 to October 2024 with no gap. Each
# entry names the Mendeley dataset id, the version to use, and the years it covers.
DATASETS = [
    {"id": "73rb78pmzw", "version": 2, "years": "2013-2020"},
    {"id": "97hkwn5y3x", "version": 4, "years": "2021"},
    {"id": "jmcdmnyh2s", "version": 2, "years": "2022"},
    {"id": "ss5pfw8xnk", "version": 1, "years": "2023-2024"},
]

ZIP_URL_TEMPLATE = "https://data.mendeley.com/public-api/zip/{dataset_id}/download/{version}"
LICENCE = "CC BY 4.0"
ATTRIBUTION = "Barack Wanjawa, University of Nairobi, via Mendeley Data"


def sha256_of_file(path):
    """The sha256 of a file's bytes, read in chunks so a large zip does not fill memory."""
    hasher = hashlib.sha256()
    with open(path, "rb") as file_handle:
        chunk = file_handle.read(65536)
        while len(chunk) > 0:
            hasher.update(chunk)
            chunk = file_handle.read(65536)
    return hasher.hexdigest()


def download_zip(dataset_id, version, destination_path):
    """One GET of the whole dataset as a zip. Mendeley needs no sign-in for this endpoint."""
    url = ZIP_URL_TEMPLATE.format(dataset_id=dataset_id, version=version)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    destination_path.write_bytes(response.content)
    return url


def extract_zip(zip_path, extract_dir):
    """Unzip every file into extract_dir directly, ignoring any subfolder the zip used."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted_files = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            member_name = Path(member.filename).name  # drop any folder the zip wrapped it in
            target_path = extract_dir / member_name
            with archive.open(member) as source, open(target_path, "wb") as target:
                target.write(source.read())
            extracted_files.append(target_path)
    return extracted_files


def count_csv_rows(csv_path):
    """Row count excluding the header, by counting lines -- no parsing needed yet."""
    with open(csv_path, "rb") as file_handle:
        line_count = sum(1 for _ in file_handle)
    return max(line_count - 1, 0)


def main():
    refresh = "--refresh" in sys.argv
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {"licence": LICENCE, "attribution": ATTRIBUTION, "datasets": {}}

    for dataset in DATASETS:
        dataset_id = dataset["id"]
        version = dataset["version"]
        zip_path = RAW_DIR / f"{dataset_id}.zip"
        extract_dir = RAW_DIR / dataset_id

        if zip_path.exists() and not refresh:
            print(f"{dataset_id} ({dataset['years']}): using cached {zip_path.name}")
            source_url = ZIP_URL_TEMPLATE.format(dataset_id=dataset_id, version=version)
        else:
            print(f"{dataset_id} ({dataset['years']}): downloading")
            source_url = download_zip(dataset_id, version, zip_path)

        extracted_files = extract_zip(zip_path, extract_dir)
        files_info = []
        for file_path in sorted(extracted_files):
            row_count = count_csv_rows(file_path)
            files_info.append({
                "file": file_path.name,
                "bytes": file_path.stat().st_size,
                "rows_excluding_header": row_count,
            })
            print(f"  {file_path.name}: {row_count} rows")

        manifest["datasets"][dataset_id] = {
            "years": dataset["years"],
            "version": version,
            "source_url": source_url,
            "zip_sha256": sha256_of_file(zip_path),
            "zip_bytes": zip_path.stat().st_size,
            "files": files_info,
        }

    manifest_path = RAW_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {manifest_path}")


if __name__ == "__main__":
    main()
