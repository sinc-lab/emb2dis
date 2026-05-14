# emb2dis: protein disorder prediction tool
This repository contains a deep learning tool for predicting intrinsically disordered regions (IDRs) in protein sequences. 

This tool generates embeddings from raw protein sequences using a pre-trained protein language model (pLM) and predicts disorder probabilities using a deep learning model that was trained with the **DisProt dataset** (2023_12) and tested on the **CAID3v3** benchmarks. The output of the tool includes per-residue disorder scores, plots of disorder along the sequence and summary statistics.

![emb2dis](emb2dis.png)

## Environment setup

1. **Clone the repository:**
```bash
git clone https://github.com/sofiaaduarte/emb2dis.git
cd emb2dis
```

2. **Create a virtual environment**:
```bash
conda create -n emb2dis python=3.11
conda activate emb2dis
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```
## Usage
The main script is `predict_disorder.py`. You can provide a FASTA file containing one or more protein sequences:
```
python predict_disorder.py --fasta data/samples.fasta
```

This script will:
- Read all sequences from the FASTA file.
- Generate embeddings using the specified pLM (ProtT5 by default).
- Predict disorder scores for each residue using a sliding window approach.
- Save results (CSV and plots) to the output directory (`./results/` by default).
- Print disorder statistics to the console.

### Command-line Arguments
| Argument | Short | Description |
|----------|-------|-------------|
| `--fasta` | `-f` | Path to input FASTA file (required). |
| `--model` | `-m` | Protein language model: `ProtT5` (by default) or `ESM2` |
| `--output-dir` | `-o` | Directory to save predictions (.csv) and plots (.png) (`./results/` by default). |
| `--device` | `-d` | Device: `cpu`, `cuda` (by default), `cuda:0`, etc. |
| `--verbose` | `-v` | Enable verbose output for detailed progress (`False` by default). |

### Examples
**1. Specify output directory and verbose mode:**
```
python predict_disorder.py --fasta data/samples.fasta --output-dir my_results/ --verbose
```
**2. Use ESM2 model on CPU:**
```
python predict_disorder.py --fasta data/samples.fasta --model ESM2 --device cpu
```
**3. Use a specific GPU:**
```
python predict_disorder.py --fasta data/samples.fasta --device cuda:1
```
## Models
### Supported Protein Language Models

| Model | Description | Embedding Size | Reference | Repository |
|-------|-------------|----------------|-----------|------------|
| **ESM2** | ESM-2 (650M parameters) | 1280 | [Lin et al., 2023](https://doi.org/10.1126/science.ade2574) | [facebookresearch/esm](https://github.com/facebookresearch/esm) |
| **ProtT5** | ProtT5-XL (half precision) | 1024 | [Elnaggar et al., 2021](https://doi.org/10.1109/TPAMI.2021.3095381) | [rostlab/ProtTrans](https://github.com/rostlab/ProtTrans) |

The disorder prediction models are trained specifically for each pLM. 

Additional models will be added in future releases.

### Additional notes
- **Sequence preprocessing**: Non-canonical amino acids (U, Z, O, B, J) are automatically converted to 'X' before generating embeddings.

## Container (CAID challenge)

For this usage, embeddings need to be pre-computed, one .npy file per sequence, outside the container and mounted in at runtime.

Inside the docker  image we have the trained classifier in a a CPU build with a  minimal set of dependencies. It does not has access to internet in prediction time. 

### 1. Pre-compute ProtT5 embeddings (host side)

```bash
python scripts/precompute_prott5.py \
  --fasta data/samples.fasta \
  --output-dir data/embeddings/
```

This writes one `{protein_id}.npy` per FASTA record (shape `(1024, L)`).

### 2. Run the container

The image is published on Docker Hub. Pull it once:

```bash
docker pull lbugnon/emb2dis:caid
```

Then run it offline with a FASTA + pre-computed embeddings:

```bash
docker run --rm --network none \
  -v user_fasta_path/samples.fasta:/data/input.fasta:ro \
  -v user_emb_path:/data/embeddings:ro \
  -v user_output_path:/data/output \
  lbugnon/emb2dis:caid --threads 4
```

The container-side paths are fixed by the image; the host paths on the left of each `:` can be anything (**note** check that paths should be absolute or relative prepending ./). Mount three host paths into:
- `/data/input.fasta`: the FASTA file (read only).
- `/data/embeddings/`: one `{protein_id}.npy` per FASTA record (from step 1, read only).
- `/data/output/`: where results are written.

Output layout:
- `/data/output/{protein_id}.caid` — one file per protein, CAID format.
- `/data/output/timings.csv` — per-sequence execution time in milliseconds.

### 3. (Optional) Build and publish Docker Hub

We already provide the image in DockerHub. To build the image from this repo:

```bash
docker build --network=host -t emb2dis:caid .
```

To publish,
```bash
docker login
docker tag emb2dis:caid <dockerhub-user>/emb2dis:caid
docker push <dockerhub-user>/emb2dis:caid
```