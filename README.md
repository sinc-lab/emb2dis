# e_emb2dis: protein disorder prediction tool based on an ensemble of emb2dis models

This repository contains a deep learning tool for predicting intrinsically disordered regions (IDRs) in protein sequences. The tool uses an ensemble of pre-trained emb2dis models to predict residue-level disorder scores from precomputed ESM-2 embeddings and returns the results in CAID-style format.

The disorder prediction models were trained using disorder annotations from DisProt (release 2025_12) as positive examples and observed residues from MobiDB as negatives.

## Local environment setup

1. **Clone the repository:**
```bash
git clone --branch e_emb2dis_caid4 --single-branch https://github.com/sinc-lab/emb2dis.git
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
The main prediction script is `predict_disorder.py`. It takes as input a FASTA file and a directory containing precomputed ESM-2 embeddings, with one `{protein_id}.npy` file per FASTA record.

For a quick test, the repository includes a sample FASTA file and example embeddings. They can be used as follows:

```bash
python predict_disorder.py --fasta data/samples.fasta --embedding-dir data/embeddings/
```

This script will:
- Read all sequences from the FASTA file.
- Load precomputed ESM-2 embeddings from the embeddings directory.
- Predict disorder scores for each residue using a sliding window approach.
- Save CAID-format outputs under the output directory (`./results/disorder/` by default).
- Save a `timings.csv` file with per-sequence execution times.

### Command-line Arguments
| Argument | Short | Description |
|----------|-------|-------------|
| `--fasta` | `-f` | Path to input FASTA file (required). |
| `--embedding-dir` | `-e` | Directory with pre-computed embeddings (.npy), one file per FASTA record. |
| `--output-dir` | `-o` | Directory where CAID outputs are written (`./results/` by default; files go into `disorder/`). |
| `--device` | `-d` | Device to run predictions on (default: `cpu`). |
| `--threads` |  | Cap the number of CPU threads used by PyTorch (torch.set_num_threads). |
| `--verbose` | `-v` | Enable verbose output for detailed progress. |

### Examples
**1. Specify output directory and verbose mode:**
```bash
python predict_disorder.py --fasta data/samples.fasta --embedding-dir data/embeddings/ --output-dir my_results/ --verbose
```
**2. Use on a specific GPU:**
```bash
python predict_disorder.py --fasta data/samples.fasta --embedding-dir data/embeddings/ --device cuda:1
```

### Supported Protein Language Model

| Model | Description | Embedding Size | Reference | Repository |
|-------|-------------|----------------|-----------|------------|
| **ESM2** | ESM-2 (650M parameters) | 1280 | [Lin et al., 2023](https://doi.org/10.1126/science.ade2574) | [facebookresearch/esm](https://github.com/facebookresearch/esm) |


## Container usage for CAID challenge

For the CAID challenge, the Docker container performs only the disorder prediction step. ESM-2 embeddings must be computed beforehand on the host machine and mounted into the container at runtime, one `{protein_id}.npy` per FASTA record.

The Docker image includes the trained disorder prediction models in a CPU build with a minimal set of dependencies. The container is intended to run without internet access during prediction.

### 1. Pre-compute ESM-2 embeddings on the host machine

If the embeddings are not available, they can be generated on the host machine before running the container:

```bash
python scripts/compute_esm2.py \
  --fasta data/samples.fasta \
  --output-dir data/embeddings/
```

This writes one `{protein_id}.npy` per FASTA record (shape `(1280, L)`, where `L` is the sequence length).

### 2. Pull the Docker image and run the container

The image is available on Docker Hub:

```bash
docker pull sofiaaduarte/e_emb2dis:caid4
```

Run the container by mounting three paths from the host machine: the input FASTA file, the directory containing the precomputed embeddings, and the output directory.

```bash
docker run --rm --network none \
  -v <path_to_fasta>:/data/input.fasta:ro \
  -v <path_to_embeddings>:/data/embeddings:ro \
  -v <path_to_output>:/data/output \
  sofiaaduarte/e_emb2dis:caid4 --threads 8
```

The paths on the left side of each `:` correspond to files or directories on the host machine and can be changed by the user. The paths on the right side are fixed inside the container and must remain unchanged.

| Host path              | Container path      | Description                                                                              |
| ---------------------- | ------------------- | ---------------------------------------------------------------------------------------- |
| `<path_to_fasta>`      | `/data/input.fasta` | Input FASTA file, mounted as read-only.                                                  |
| `<path_to_embeddings>` | `/data/embeddings`  | Directory containing one `{protein_id}.npy` file per FASTA record, mounted as read-only. |
| `<path_to_output>`     | `/data/output`      | Directory where prediction files are written.                                            |


The container can be tested with the sample files included in this repository:

```bash
docker run --rm --network none \
  -v "$(pwd)/data/samples.fasta:/data/input.fasta:ro" \
  -v "$(pwd)/data/embeddings:/data/embeddings:ro" \
  -v "$(pwd)/results:/data/output" \
  sofiaaduarte/e_emb2dis:caid4 \
  --threads 8
```

Output layout:
- `/data/output/{protein_id}.caid`: one file per protein, CAID format.
- `/data/output/timings.csv`: per-sequence execution time in milliseconds.

### 3. (Optional) Build and publish Docker image

To build the image from this repository, run the following command:

```bash
docker build --network=host -t e_emb2dis:caid4 .
```

To publish,
```bash
docker login
docker tag e_emb2dis:caid4 <dockerhub-user>/e_emb2dis:caid4
docker push <dockerhub-user>/e_emb2dis:caid4
```