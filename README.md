# P-GTNS: Protein Graph Transformer Network for Protein Function Prediction

## Project Overview

This project constructs protein graphs based on sequence and structural information and uses a Graph Transformer Network for protein function prediction.
It supports generating graph data from PDB or FASTA files, as well as training, testing, and predicting with the model.

## Project Structure

```
P-GTNS/
│
├─ data/processed/        # Processed sequence or structure files
├─ graph_embding.py       # Graph construction and related functions
├─ train.py               # Training script
├─ test.py                # Testing script
├─ predictor.py           # Prediction script
└─ requirements.txt       # Python dependencies
```

## Install Dependencies

Install the required Python packages first:

```bash
pip install -r requirements.txt
```

## Data Preparation

Place protein sequences or PDB files into the `data/processed/` folder.
Place the GO annotation file in the `data/` directory, for example:

```
data/nrPDB-GO_2024.06.24_annot.tsv
```

## Training the Model

Use `train.py` to train the model and generate weights:

```bash
python train.py
```

## Testing the Model

Use `test.py` to evaluate the trained model:

```bash
python test.py
```

## Making Predictions

Use `predictor.py` to make predictions on new protein data:

```bash
python predictor.py
```

## Notes

* Input file formats must be correct:

  * PDB files are used for generating 3D structure graphs
  * FASTA files can only generate sequence adjacency graphs
* GPU with CUDA is recommended; otherwise, CPU can be used for training
* ESM model weights should be placed in `model_weight/esm1b.pt`

