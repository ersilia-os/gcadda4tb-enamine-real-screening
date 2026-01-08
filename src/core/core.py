import os
import numpy as np
import joblib
import json
import time
import h5py
import collections
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import normalize

from .utils import get_filenames, get_endpoints_dir


root = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(root, "..", "processed", "20_billion_scale_prototype")


GARDP_ENDPOINTS = [
    "abaumannii_perc01_full",
    "kpneumoniae_nctc_13438_perc01_full",
    "kpneumoniae_atcc_43816_perc01_full",
    "kpneumoniae_ecl8_perc05_full",
    "abaumannii_perc01_attractive",
    "kpneumoniae_nctc_13438_perc01_attractive",
    "kpneumoniae_atcc_43816_perc05_attractive",
    "kpneumoniae_ecl8_perc05_attractive",
]

OTHER_ACTIVITY_ENDPOINTS = [
    "ecoli_stokes",
    "abaumannii_stokes",
    "gneprop_hts",
    "gn_class",
    "mole_gn",
    "bcenocepacia"
]

TRANSPORT_ENDPOINTS = [
    "entry_rules",
    "retention_ecoli",
    "efflux_evader_ecoli",
    "pumped_ecoli",
    "permeability_proxy_ecoli",
    "permeability_paeruginosa"
]

TOXICITY_ENDPOINTS = [
    "cytotoxicity",
]

ATTRACTIVENESS_ENDPOINTS = [
    "antibioticdb",
    "collins_abx",
    "abx_resemblance",
    "pains",
    "frequent_hitters",
]

ALL_ENDPOINTS = (
    GARDP_ENDPOINTS +
    OTHER_ACTIVITY_ENDPOINTS +
    TRANSPORT_ENDPOINTS +
    TOXICITY_ENDPOINTS +
    ATTRACTIVENESS_ENDPOINTS
)


class ScreenerArtifact(object):
    
    def __init__(self, endpoint, model, data, h5_probas_file, h5_passed_file):
        self.endpoint = endpoint
        self.model = model
        self.keep = data["keep"]
        self.cutoff = data["cutoff"]
        self.min_keep_proportion = 0.001
        self.max_keep_proportion = 0.5
        self.min_discard_proportion = 0.5
        self.max_discard_proportion = 0.9
        self.h5_probas = h5_probas_file
        self.h5_pass = h5_passed_file
        self.is_first = True

    def add_predict_proba_batch(self, X):
        y_hat = self.model.predict_proba(X)[:, 1]
        with h5py.File(self.h5_probas, "a") as f:
            if self.is_first:
                if self.endpoint in f:
                    del f[self.endpoint]
                maxshape = (None,)
                f.create_dataset(self.endpoint, data=y_hat, maxshape=maxshape, chunks=True, compression="gzip")
                self.is_first = False
            else:
                dset = f[self.endpoint]
                current_size = dset.shape[0]
                new_size = current_size + y_hat.shape[0]
                dset.resize((new_size,))
                dset[current_size:new_size] = y_hat

    def screen(self):
        with h5py.File(self.h5_probas, "r") as f:
            y_hat = f[self.endpoint][:]

        if self.keep:
            y_bin = (y_hat >= self.cutoff).astype(np.int8)
            n_pass = np.sum(y_bin)
            min_pass = int(len(y_hat) * self.min_keep_proportion)
            max_pass = int(len(y_hat) * self.max_keep_proportion)
            if n_pass < min_pass:
                sorted_indices = np.argsort(y_hat)[::-1]
                y_bin[:] = 0
                y_bin[sorted_indices[:min_pass]] = 1
            if n_pass > max_pass:
                sorted_indices = np.argsort(y_hat)[::-1]
                y_bin[:] = 0
                y_bin[sorted_indices[:max_pass]] = 1
            else:
                pass
        else:
            y_bin = (y_hat <= self.cutoff).astype(np.int8)
            n_pass = np.sum(y_bin)
            min_pass = int(len(y_hat) * self.min_discard_proportion)
            max_pass = int(len(y_hat) * self.max_discard_proportion)
            if n_pass < min_pass:
                sorted_indices = np.argsort(y_hat)
                y_bin[:] = 0
                y_bin[sorted_indices[:min_pass]] = 1
            if n_pass > max_pass:
                sorted_indices = np.argsort(y_hat)
                y_bin[:] = 0
                y_bin[sorted_indices[:max_pass]] = 1
            else:
                pass
        
        with h5py.File(self.h5_pass, "a") as f:
            if self.endpoint in f:
                del f[self.endpoint]
            f.create_dataset(self.endpoint, data=y_bin, dtype=np.int8, compression="gzip")

        print(f"Endpoint {self.endpoint}: Passed {np.sum(y_bin)} out of {len(y_bin)} compounds. Strict passes = {n_pass}.")


def fps_iterator(h5_file, chunksize=500_000):
    with h5py.File(h5_file, "r") as h5_file:
        n_fps = h5_file["values"].shape[0]
        for start in range(0, n_fps, chunksize):
            end = min(start + chunksize, n_fps)
            fps = h5_file["values"][start:end]
            smiles_list = h5_file["key"][start:end].astype(str).tolist()
            identifiers = h5_file["input"][start:end].astype(str).tolist()
            yield fps, smiles_list, identifiers


def load_endpoints(dir_path, chunk_name):
    file_names = get_filenames(dir_path, chunk_name)
    h5_probas_file = file_names["h5_probas_file"]
    h5_passed_file = file_names["h5_passed_file"]
    endpoints_dir = get_endpoints_dir()
    endpoints = []
    for endpoint in ALL_ENDPOINTS:
        model = joblib.load(os.path.join(endpoints_dir, f"{endpoint}.joblib"))
        with open(os.path.join(endpoints_dir, f"{endpoint}.json"), "r") as f:
            data = json.load(f)
        endpoints += [ScreenerArtifact(endpoint, model, data, h5_probas_file, h5_passed_file)]
    return endpoints


def screen(dir_path, chunk_name, endpoints):
    file_names = get_filenames(dir_path, chunk_name)
    h5_file = file_names["h5_file"]
    h5_passed_file = file_names["h5_passed_file"]
    with h5py.File(h5_file, "r") as f:
        print("Working on FP file of shape:", f["values"].shape)
        print("Identifiers sample:", f["key"][:5])
        print("SMILES sample:", f["input"][:5])

    smiles_list = []
    identifiers_list = []
    for i, (fps, smis, ids) in enumerate(fps_iterator(h5_file)):
        X_norm = normalize(fps, norm="l2")
        for endpoint in endpoints:
            print("Processing endpoint:", endpoint.endpoint, "on batch", i, "of size", fps.shape)
            endpoint.add_predict_proba_batch(X_norm)
        smiles_list += smis
        identifiers_list += ids
    endpoint_names = []
    for i, endpoint in enumerate(endpoints):
        endpoint_names += [endpoint.endpoint]
        endpoint.screen()

    with h5py.File(h5_passed_file, "a") as f:
        if "endpoints" in f:
            del f["endpoints"]
        if "shape" in f:
            del f["shape"]
        f.create_dataset("endpoints", data=np.array(endpoint_names, dtype=h5py.string_dtype()))
        f.create_dataset("shape", data=np.array([len(smiles_list), len(endpoint_names)], dtype=np.int64))


def filter(output_dir, chunk_name):

    file_names = get_filenames(output_dir, chunk_name)
    h5_file = file_names["h5_file"]

    pass_proportion = 0.001

    h5_passed_file = file_names["h5_passed_file"]
    with h5py.File(h5_passed_file, "r") as f:
        endpoints = f["endpoints"][:].astype(str).tolist()
        shape = tuple(f["shape"][:])

    Y = np.zeros(shape, dtype=np.int8)
    with h5py.File(h5_passed_file, "r") as f:
        for i, endpoint in enumerate(endpoints):
            Y[:, i] = f[endpoint][:]
    
    n_pass = int(pass_proportion * Y.shape[0])
    min_n_pass = int(max(n_pass / 2, 1))
    max_n_pass = int(max(n_pass * 2, 2))

    print(f"Filtering to get between {min_n_pass} and {max_n_pass} compounds passing the screening pipeline.")

    print("Block 1. Attractiveness and toxicity filtering...")
    columns = [
        "cytotoxicity",
        "antibioticdb",
        "collins_abx",
        "abx_resemblance",
        "pains",
        "frequent_hitters",
    ]
    col_idxs = [endpoints.index(e) for e in columns]
    passes = np.sum(Y[:, col_idxs], axis=1)
    pass_mask_b1 = (passes >= 5)
    print(f"Compounds passing attractiveness and toxicity filters: {np.sum(pass_mask_b1)} out of {Y.shape[0]}")

    print("Block 2. GARDP activity filters for resistant strains...")
    columns = [
        "abaumannii_perc01_full",
        "kpneumoniae_nctc_13438_perc01_full",
        "abaumannii_perc01_attractive",
        "kpneumoniae_nctc_13438_perc01_attractive",
    ]
    col_idxs = [endpoints.index(e) for e in columns]
    passes = np.sum(Y[:, col_idxs], axis=1)
    pass_mask_b2 = (passes >= 1)
    print(f"Compounds passing GARDP resistant strains filters: {np.sum(pass_mask_b2)} out of {Y.shape[0]}")

    print("Block 3. Activity filters for other strains, species and datasets...")
    columns = [
        "kpneumoniae_atcc_43816_perc01_full",
        "kpneumoniae_ecl8_perc05_full",
        "kpneumoniae_atcc_43816_perc05_attractive",
        "kpneumoniae_ecl8_perc05_attractive",
        "ecoli_stokes",
        "abaumannii_stokes",
        "gneprop_hts",
        "gn_class",
        "mole_gn",
        "bcenocepacia"
    ]
    col_idxs = [endpoints.index(e) for e in columns]
    passes = np.sum(Y[:, col_idxs], axis=1)
    pass_mask_b3 = (passes >= 2)
    print(f"Compounds passing other activity filters: {np.sum(pass_mask_b3)} out of {Y.shape[0]}")

    print("Block 4. Transport filters...")
    columns = [
        "entry_rules",
        "retention_ecoli",
        "efflux_evader_ecoli",
        "pumped_ecoli",
        "permeability_proxy_ecoli",
        "permeability_paeruginosa"
    ]
    col_idxs = [endpoints.index(e) for e in columns]
    passes = np.sum(Y[:, col_idxs], axis=1)
    pass_mask_b4 = (passes >= 3)
    print(f"Compounds passing transport filters: {np.sum(pass_mask_b4)} out of {Y.shape[0]}")

    print("Combining all filters...")
    pass_mask = pass_mask_b1 & (pass_mask_b2 | (pass_mask_b3 & pass_mask_b4))
    print(f"Total compounds passing the default screening pipeline: {np.sum(pass_mask)} out of {Y.shape[0]}")

    def auxiliary_score():
        print("Computing auxiliary scores for relaxation...")
        columns = [
            "abaumannii_perc01_full",
            "kpneumoniae_nctc_13438_perc01_full",
            "kpneumoniae_atcc_43816_perc01_full",
            "abaumannii_perc01_attractive",
            "kpneumoniae_nctc_13438_perc01_attractive",
        ]
        scores = np.zeros((Y.shape[0], ), np.float32)
        h5_probas_file = file_names["h5_probas_file"]
        with h5py.File(h5_probas_file, "r") as f:
            for e in columns:
                scores += f[e][:]
        print("Auxiliary scores computed:", scores)
        print("Max score:", np.max(scores))
        print("Min score:", np.min(scores))
        print("Shape of scores:", scores.shape)
        return scores

    scores = auxiliary_score()

    if np.sum(pass_mask) < min_n_pass:
        used_indices = set(np.where(pass_mask)[0])
        orig_indices = used_indices.copy()
        print("Not enough compounds passed. Relaxing criteria...")
        sorted_indices = np.argsort(scores)[::-1]
        sorted_indices = [i for i in sorted_indices if i not in used_indices]
        n_needed = int(min_n_pass - len(used_indices))
        print(f"Need to add {n_needed} more compounds to meet the minimum pass requirement.")
        cytotoxic_idxs = set(np.where(Y[:, endpoints.index("cytotoxicity")] == 1)[0])
        for idx in sorted_indices:
            if idx in cytotoxic_idxs:
                continue
            pass_mask[idx] = True
            used_indices.add(idx)
            n_needed -= 1
            if n_needed <= 0:
                break
        if n_needed > 0:
            print("Warning: Could not find enough non-cytotoxic compounds to meet the minimum pass requirement.")
            sorted_indices = [i for i in sorted_indices if i not in used_indices]
            for idx in sorted_indices[:n_needed]:
                pass_mask[idx] = True
        passed_indices = set(np.where(pass_mask)[0])
        assert orig_indices.issubset(passed_indices), "Some originally passing indices were lost"
        print(f"After adjustment, total compounds passing: {np.sum(pass_mask)} out of {Y.shape[0]}")
        
    elif np.sum(pass_mask) > max_n_pass:
        print("Too many compounds passed. Tightening criteria...")
        used_indices = set(np.where(pass_mask)[0])
        sorted_indices = np.argsort(scores)
        sorted_indices = [i for i in sorted_indices if i in used_indices]
        cytotoxic_idxs = set(np.where(Y[:, endpoints.index("cytotoxicity")] == 1)[0])
        kept_indices = set([i for i in sorted_indices if i not in cytotoxic_idxs][:int(max_n_pass)])
        if len(kept_indices) < int(max_n_pass):
            n_needed = int(max_n_pass - len(kept_indices))
            for idx in sorted_indices:
                if idx in kept_indices:
                    continue
                kept_indices.add(idx)
                n_needed -= 1
                if n_needed <= 0:
                    break
        kept_indices = sorted(kept_indices)
        pass_mask[:] = False
        pass_mask[kept_indices] = True
        assert len(set(used_indices).intersection(kept_indices)) == len(kept_indices), "Mismatch in kept indices"
        print(f"After adjustment, total compounds passing: {np.sum(pass_mask)} out of {Y.shape[0]}")

    else:
        print("Number of passing compounds within desired range. No adjustment needed.")
        pass

    with h5py.File(h5_file, "r") as f:
        smiles = f["input"][:].astype(str).tolist()
        identifiers = f["key"][:].astype(str).tolist()
        filtered_smiles = [smiles[i] for i in range(len(smiles)) if pass_mask[i]]
        filtered_ids = [identifiers[i] for i in range(len(identifiers)) if pass_mask[i]]

    data = collections.OrderedDict()
    data["smiles"] = filtered_smiles
    data["identifier"] = filtered_ids
    data["auxiliary_score"] = scores[pass_mask]
    Y = Y[pass_mask, :]
    for i, endpoint in enumerate(endpoints):
        data[endpoint] = Y[:, i]
    
    hits_csv = file_names["hits_csv"]
    df = pd.DataFrame(data)
    df.to_csv(hits_csv, index=False)
    print(f"Filtered compounds saved to {hits_csv}.")
    
    return df


def stats(output_dir, chunk_name):
    file_names = get_filenames(output_dir, chunk_name)
    h5_pass_file = file_names["h5_passed_file"]
    with h5py.File(h5_pass_file, "r") as f:
        endpoints = f["endpoints"][:].astype(str).tolist()
        shape = tuple(f["shape"][:])
        data = collections.OrderedDict()
        for endpoint in endpoints:
            y_bin = f[endpoint][:]
            n_pass = int(np.sum(y_bin))
            data[endpoint] = n_pass
    output_json = file_names["stats_json"]
    with open(output_json, "w") as f:
        data = {
            "n_compounds": int(shape[0]),
            "n_endpoints": int(shape[1]),
            "passes_per_endpoint": data
        }
        json.dump(data, f, indent=4)
    print(f"Statistics saved to {output_json}.")
    print(data)


def pipeline(output_dir, chunk_name):
    endpoints = load_endpoints(output_dir, chunk_name)
    screen(output_dir, chunk_name, endpoints)
    filter(output_dir, chunk_name)
    stats(output_dir, chunk_name)
    