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
    # print(f"Downloading identifiers_file to {identifiers_file}...")
    # download_file(identifiers_file)
    print(f"Downloading data_file to {data_file}...")
    download_file(data_file)
    t1 = time.time()
    print(f"Download completed in {t1 - t0:.2f} seconds.")


def clean_data(dir_path, chunk_name):
    print("Cleaning up intermediate file...")
    all_files = [f"{chunk_name}_X.npz"]
    for v in all_files:
        print(f"Removing {v}...")
        os.remove(os.path.join(dir_path, v))