# DGRN

This repository provides an executable implementation of DGRN. The current code runs one training/evaluation loop at a time:

1. load one processed wireless traffic dataset,
2. train the DGRN model with a learnable latent graph,
3. report validation/test forecasting accuracy.

The `data/` directory contains processed Milan (Telecom Internet), Finland (Urban Wi-Fi), and Taiwan (Vehicular GCT) subsets used by the example configs. The loaders use the files and `time_range` values specified in the YAML configs directly. 

The default configs use an 80%-10%-10% chronological split, historical window 6, prediction horizon 3, `B=3`, hidden dimension 64, temporal embedding dimension 64, `K_i=3`, `K_r=3`, inner AdamW learning rate `1e-3`, weight decay `1e-4`, outer AdamW learning rate `0.2`, `S=3` inner steps per epoch, EMA decay `0.9`, Armijo backtracking factor `0.8` (`outer_lr_backtracking_factor`), maximum 200 epochs, and early-stopping patience 20.

## Files

- `train.py`: bilevel training and evaluation script with inner model-parameter updates, once-per-epoch latent-graph updates, EMA-smoothed hypergradients, clipping, Armijo backtracking, and recurrent-tap spectral projection.
- `dgrn/model.py`: the DGRN implementation, including physical graph filtering, dense entmax latent graph learning, recurrent graph blocks, and forecast head.
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

Outputs are saved under `runs/<dataset>/`, including:

- `best_model.pt`
- `learned_adj.csv`
- `metrics.json`

## DGRN multi-run evaluation

To run multiple independent seeds for all three datasets and report the mean and
sample standard deviation of RMSE, MAE, and R$^2$:

```bash
python scripts/multiple_run_evaluation.py --device cuda
```

The command writes each trial's `metrics.json`, logs, a
`per_run_metrics.csv` and `summary.csv`, and `dgrn_multi_run.md` under
`runs_multiple_evaluation/`.

The seeds used in paper for 10 runs are set as default: 
`873162450`, `815294637`, `425917638`, `314857206`, `841903276`,
`168935742`, `782410593`, `638274591`, `183649570`, and `350761928`.
Standard deviations use the sample definition (`n-1`).
