# Call ensure_all_downloaded() when needed

from pathlib import Path
from datetime import datetime
import json
import subprocess
import requests
import shutil

MARKER_NAME = ".dataset_ready"

DATASETS = [
    {
        "name": "our_dataset",
        "path": Path("data/raw/our_dataset"),
        "type": "gdrive",
        "folder_id": "1o9KxK52oGi1hZoTS82DmheeAk2P6MR10",
        "source": "gdrive:our_dataset",
    },
    {
        "name": "our_bot_dataset",
        "path": Path("data/raw/our_bot_dataset"),
        "type": "gdrive",
        "folder_id": "13wx3vIMZ87HQBLQJx5xSQROT4uVVdbP-",
        "source": "gdrive:our_bot_dataset",
    },
    {
        "name": "boun",
        "path": Path("data/raw/boun-mouse-dynamics-dataset"),
        "type": "url",
        "url": "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/w6cxr8yc7p-2.zip",
        "source": "s3:boun-dataset",
    },
]

def _marker_path(folder: Path) -> Path:
    return folder / MARKER_NAME

def is_dataset_ready(folder: Path) -> bool:
    return _marker_path(folder).exists()

def mark_dataset_ready(folder: Path, source: str):
    marker = {
        "source": source,
        "ready_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(_marker_path(folder), "w") as f:
        json.dump(marker, f, indent=2)

def download_gdrive_folder(folder_id: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "gdown",
            "--folder",
            f"https://drive.google.com/drive/folders/{folder_id}",
            "-O",
            str(output_dir),
        ],
        check=True,
    )


def extract_multipart_zip(zip_path: Path):
    zip_path = zip_path.resolve()
    dest_dir = zip_path.parent

    print(f"[i] Extracting multipart archive in {dest_dir}")

    result = subprocess.run(
        ["7z", "x", zip_path.name, "-y"],
        cwd=dest_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

def verify_boun_extracted(dest_dir: Path):
    expected = dest_dir / "users"
    if not expected.exists():
        raise RuntimeError("BOUN extraction failed: 'users/' directory not found")

def download_and_extract_boun(url: str, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dest_dir / "boun-mouse-dynamics-dataset.zip"

    if not zip_path.exists():
        print("[↓] Downloading BOUN archive...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    extract_multipart_zip(zip_path)

    force_flatten_boun(dest_dir)

def force_flatten_boun(dest_dir: Path):
    nested = dest_dir / dest_dir.name

    if not nested.exists():
        print("[✓] BOUN already flat")
        return

    print("[!] Flattening nested BOUN directory")

    for item in nested.iterdir():
        target = dest_dir / item.name
        if target.exists():
            continue
        shutil.move(str(item), target)

    shutil.rmtree(nested)
    print("[✓] BOUN flattened successfully")

def ensure_all_downloaded():
    for ds in DATASETS:
        path = ds["path"]

        if is_dataset_ready(path):
            print(f"[✓] {ds['name']} ready")
            continue

        print(f"[↓] Downloading {ds['name']}...")
        path.mkdir(parents=True, exist_ok=True)

        if ds["type"] == "gdrive":
            download_gdrive_folder(ds["folder_id"], path)

        elif ds["type"] == "url":
            download_and_extract_boun(ds["url"], path)

        else:
            raise ValueError(f"Unknown dataset type: {ds['type']}")

        mark_dataset_ready(path, ds["source"])
        print(f"[✓] {ds['name']} downloaded")
    
