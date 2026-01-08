import os
import io
import sys
import h5py
import time
import httplib2
import pandas as pd
import numpy as np
from tqdm import tqdm
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.errors import HttpError

GDRIVE_FOLDER_ID = "1FBELagBf9hlKVgvkaZ8YF60jKRAmsHPo"
ARRAY_DTYPE = np.int8


def _get_data_file(dir_path, chunk_name):
    return os.path.join(dir_path, f"{chunk_name}_X.npz")


def _get_identifiers_file(dir_path, chunk_name):
    return os.path.join(dir_path, f"{chunk_name}_SMILES_IDs.tsv.zip")


def _get_h5_file(dir_path, chunk_name):
    return os.path.join(dir_path, f"{chunk_name}.h5")


def _get_probas_h5_file(dir_path, chunk_name):
    return os.path.join(dir_path, f"{chunk_name}_probas.h5")


def _get_passed_h5_file(dir_path, chunk_name):
    return os.path.join(dir_path, f"{chunk_name}_passed.h5")


def _get_hits_csv(dir_path, chunk_name):
    return os.path.join(dir_path, f"{chunk_name}_hits.csv")


def _get_stats_json(dir_path, chunk_name):
    return os.path.join(dir_path, f"{chunk_name}_stats.json")


def get_filenames(dir_path, chunk_name):
    data_file = _get_data_file(dir_path, chunk_name)
    identifiers_file = _get_identifiers_file(dir_path, chunk_name)
    h5_file = _get_h5_file(dir_path, chunk_name)
    hits_csv = _get_hits_csv(dir_path, chunk_name)
    stats_json = _get_stats_json(dir_path, chunk_name)
    h5_probas_file = _get_probas_h5_file(dir_path, chunk_name)
    passed_h5_file = _get_passed_h5_file(dir_path, chunk_name)

    results = {
        "data_file": data_file,
        "identifiers_file": identifiers_file,
        "h5_file": h5_file,
        "hits_csv": hits_csv,
        "stats_json": stats_json,
        "h5_probas_file": h5_probas_file,
        "h5_passed_file": passed_h5_file,
    }

    return results


def get_endpoints_dir():
    root = os.path.dirname(os.path.abspath(__file__))
    endpoints_dir = os.path.abspath(os.path.join(root, "..", "..", "data", "endpoints"))
    return endpoints_dir


def check_exists(dir_path, chunk_name):
    data_file = _get_data_file(dir_path, chunk_name)
    identifiers_file = _get_identifiers_file(dir_path, chunk_name)
    if os.path.exists(data_file) and os.path.exists(identifiers_file):
        return True
    else:
        return False


def download_file(outfile):
    root = os.path.dirname(os.path.abspath(__file__))
    file = os.path.basename(outfile)
    service_file = os.path.abspath(os.path.join(root, "..", "..", "config", "service.json"))
    folder_id = GDRIVE_FOLDER_ID
    creds = Credentials.from_service_account_file(service_file, scopes=["https://www.googleapis.com/auth/drive.readonly"])
    for attempt in range(10):
        try:
            http = httplib2.Http(timeout=600)
            authed_http = AuthorizedHttp(creds, http=http)
            service = build("drive", "v3", http=authed_http)
            query = f"name='{file}' and '{folder_id}' in parents and trashed=false"
            results = service.files().list(q=query, fields="files(id)", supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
            break
        except (HttpError, OSError):
            time.sleep(5)
            if attempt == 9:
                raise
    files = results.get("files", [])
    if not files:
        raise FileNotFoundError(f"'{file}' not found in folder {folder_id}. Consider checking available chunks in data/chunks/chunks.csv")
    if len(files) > 1:
        raise RuntimeError(f"Multiple files named '{file}' are found...")
    file_id = files[0]["id"]
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with io.FileIO(outfile, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request,chunksize=100 * 1024 * 1024)  # 100 MB chunks
        done = False
        retries = 0
        while not done:
            try:
                status, done = downloader.next_chunk()
                if status:
                    print(f"Download {int(status.progress() * 100)}%\n")
            except (HttpError, OSError):
                retries += 1
                print(f"Error found when downloading file. Trying again...[{retries}/10]")
                if retries >= 10:
                    raise
                time.sleep(10)


def download_data(dir_path, chunk_name):
    t0 = time.time()
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    data_file = _get_data_file(dir_path, chunk_name)
    identifiers_file = _get_identifiers_file(dir_path, chunk_name)
    if os.path.exists(data_file):
        os.remove(data_file)
    if os.path.exists(identifiers_file):
        os.remove(identifiers_file)
    print(f"Downloading identifiers_file to {identifiers_file}...")
    download_file(identifiers_file)
    print(f"Downloading data_file to {data_file}...")
    download_file(data_file)
    t1 = time.time()
    print(f"Download completed in {t1 - t0:.2f} seconds.")


def copy_data(from_dir, output_dir, chunk_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_file_src = _get_data_file(from_dir, chunk_name)
    identifiers_file_src = _get_identifiers_file(from_dir, chunk_name)
    data_file_dst = _get_data_file(output_dir, chunk_name)
    identifiers_file_dst = _get_identifiers_file(output_dir, chunk_name)
    print(f"Copying {data_file_src} to {data_file_dst}...")
    os.system(f"cp {data_file_src} {data_file_dst}")
    print(f"Copying {identifiers_file_src} to {identifiers_file_dst}...")
    os.system(f"cp {identifiers_file_src} {identifiers_file_dst}")
    print("Done with copying to output directory.")


def convert_to_h5(dir_path, chunk_name, batch_size=100_000):
    t0 = time.time()
    print(f"Converting chunk {chunk_name} to H5 format...")
    data_file = _get_data_file(dir_path, chunk_name)
    identifiers_file = _get_identifiers_file(dir_path, chunk_name)
    h5_file = _get_h5_file(dir_path, chunk_name)
    identifiers = pd.read_csv(identifiers_file, compression="zip", delimiter="\t")
    print(identifiers.head())
    X = np.load(data_file)["X"]
    n_rows, n_cols = X.shape
    print(f"Identifiers rows: {len(identifiers)}, Data shape: {n_rows}, {n_cols}")
    assert len(identifiers) == n_rows, "Row mismatch between X and identifiers"
    with h5py.File(h5_file, "w") as f:
        dset_X = f.create_dataset("values", shape=(n_rows, n_cols), dtype=ARRAY_DTYPE, chunks=(batch_size, n_cols), compression="gzip")
        f.create_dataset("key", data=identifiers["id"].tolist(), dtype=h5py.string_dtype())
        f.create_dataset("input", data=identifiers["smiles"].tolist(), dtype=h5py.string_dtype())
        for start in tqdm(range(0, n_rows, batch_size), desc="Converting to H5"):
            end = min(start + batch_size, n_rows)
            X_batch = X[start:end].astype(ARRAY_DTYPE)
            dset_X[start:end] = X_batch
            del X_batch
    t1 = time.time()
    print(f"H5 file saved to {h5_file}")
    print(f"Conversion completed in {t1 - t0:.2f} seconds.")
    return h5_file


def clean_data(dir_path, chunk_name):
    print("Cleaning up intermediate files...")
    all_files = get_filenames(dir_path, chunk_name)
    for _, v in all_files.items():
        if v.endswith("_hits.csv") or v.endswith("_stats.json"):
            continue
        if os.path.exists(v):
            print(f"Removing {v}...")
            os.remove(v)