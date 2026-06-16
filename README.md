# DGRN

This repository provides an executable implementation of DGRN. It runs a closed training loop:

1. load one processed wireless traffic dataset,
2. train the DGRN model with a learnable latent graph,
3. report validation/test forecasting accuracy.

The `data/` directory contains processed Milan, Finland, and Taiwan subsets used by the example configs. If you publish the code without bundling the CSV files, replace this directory with a public dataset link and keep the same filenames.

The default configs follow the paper-reported DGRN settings: 80%-10%-10% chronological split, historical window 6, prediction horizon 3, `B=3`, hidden dimension 64, temporal embedding dimension 64, `K_i=3`, `K_r=3`, AdamW learning rate `1e-3`, weight decay `1e-4`, maximum 200 epochs, and early-stopping patience 20.

## Files

- `train.py`: bilevel training and evaluation script with inner model-parameter updates and once-per-epoch latent-graph updates.
- `dgrn/model.py`: the DGRN implementation, including physical graph filtering, latent graph learning, recurrent graph blocks, and forecast head.
- `dgrn/data.py`: processed CSV loading, time features, scaling, and sliding-window datasets.
- `configs/milan.yaml`, `configs/finland.yaml`, `configs/taiwan.yaml`: runnable dataset configs.
- `data/*.csv`: processed traffic matrices and physical adjacency matrices.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Milan is the default example:

```bash
python train.py --config configs/milan.yaml
```

Other datasets:

```bash
python train.py --config configs/finland.yaml
python train.py --config configs/taiwan.yaml
```

For a quick smoke test:

```bash
python train.py --config configs/milan.yaml --epochs 1
```

The script prints a final `METRIC_JSON` line with `MAE`, `RMSE`, and `R2` in the original data scale. By default, these metrics follow the paper code's `if_missing=True` metric shape, where the reconstructed input context and the prediction horizon are concatenated before metric calculation.

Outputs are saved under `runs/<dataset>/`, including:

- `best_model.pt`
- `learned_adj.csv`
- `metrics.json`
