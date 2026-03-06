# GC-ADDA4TB Enamine REAL 10B screening

This repository facilitates the screening of Enamine REAL 10B chunks (994) using NB-based surrogate models previously trained on docking scores. 

## Background

This repository is part of the GC-ADDA4TB program to design potential degraders (BacPROTACs) that target essential tRNA synthetases in _Mycobacterium tuberculosis_.
- You can learn more about this work in this repository: https://github.com/ersilia-os/mtb-targeted-protein-degradation
- The Enamine REAL space screened was split and stored using this code: https://github.com/ersilia-os/ready-to-screen-enamine-real

## Pipeline

Code in this repository does the following:

1. Downloading a chunk of Enamine REAL containing 10M compounds alongside their ECFP6 fingerprints (already pre-calculated). These files are stored in Ersilia's Google Drive, so a service file (JSON format) is necessary to download them (see Important notes). In total, there are 994 chunks, i.e. ~10B molecules. Chunk identifiers can be found in [this file](https://github.com/ersilia-os/gcadda4tb-enamine-real-screening/blob/main/data/chunks/chunks.csv).

2. When ECFP6 are available locally, all compounds are screened against NB-based surrogate models (x276) located in `data/models` (tracked with `eosvc`). The indices associated to compounds having a predicted active probabilitiy in the 1 and 5 percentile within each model are stored in `npz` format, within the `output` directory (tracked with `eosvc` as well).

3. The downloaded file containig ECFP6s for 10M compounds is removed from the filesystem. 


## Usage

Clone this repository and create a Conda environment to install package requirements:

```bash
git clone https://github.com/ersilia-os/gcadda4tb-enamine-real-screening
cd gcadda4tb-enamine-real-screening

conda create -n gcadda4tb-enamine10b python=3.12
conda activate gcadda4tb-enamine10b
pip install -r requirements.txt
```

To run the pipeline, use the following command:

```bash
python src/run.py --chunk_name $CHUNK_NAME --output_dir $OUTPUT_DIR
```

Replace the placeholders:
- `$CHUNK_NAME`: Name of the chunk to be screened.
- `$OUTPUT_DIR`: Directory to save the output.

To download NB-based surrogate models, simply run:

```bash
eosvc download --path data
```

To download final results:

```bash
eosvc download --path output
```

## Important notes

1. It is crucial that NB-based models are trained on exactly [the same](https://github.com/ersilia-os/ready-to-screen-enamine-real/blob/main/src/src.py) RDKit version (2025.09.1) that was used to store the Enamine fingerprints in Google Drive, using the same ECFP6 count featurization (radius 3, 2048 bits).

2. To download files from Ersilia's Google Drive, please download the corresponding service file (named `service.json`) from [Platform - Shared Credentials](https://drive.google.com/drive/folders/1OPHVrMaRF_90IeQsOld74pqCsWY-s-vH) and save it in `config` under the same name.

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
