import os
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.preprocessing import normalize


root = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(root, "..", "..", "data", "models")
MODELS = sorted(os.listdir(models_dir))


def screen(output_dir, chunk_name, CHUNK_SIZE = 100_000):

    print("Loading ECFP6s...")
    X = np.load(os.path.join(output_dir, f"{chunk_name}_X.npz"))['X']
    print("ECFP6s loaded!")

    # For each model
    for model_name in tqdm(MODELS):

        # Load model
        pocket_name = model_name.replace("_nb_model.joblib", "")
        model = joblib.load(os.path.join(models_dir, model_name))

        # Predict proba in chunks
        PROBS = np.zeros(X.shape[0], dtype=np.float32)
        for i in range(0, X.shape[0], CHUNK_SIZE):

            # Normalize chunk -same as during training- and predict
            probs_chunk = model.predict_proba(normalize(X[i:i+CHUNK_SIZE], norm='l2'))[:,1]
            PROBS[i:i+len(probs_chunk)] = probs_chunk

        # Get thresholds
        thr_5 = np.percentile(PROBS, 95)
        thr_1 = np.percentile(PROBS, 99)

        # Get indices above thresholds
        ind_5 = np.where(PROBS >= thr_5)[0].astype(np.uint32)
        ind_1 = np.where(PROBS >= thr_1)[0].astype(np.uint32)

        # Save indices in npz files
        np.savez_compressed(os.path.join(output_dir, f"{pocket_name}_ind_5.npz"), ind_5=ind_5, thr=thr_5)
        np.savez_compressed(os.path.join(output_dir, f"{pocket_name}_ind_1.npz"), ind_1=ind_1, thr=thr_1)