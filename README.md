# TSF-Resonator

PyTorch implementation of **TSF-Resonator** and several baseline forecasters for multivariate wastewater time-series forecasting.

The project is designed for experiments where all available numeric process variables are used as model inputs, while selected target variables are evaluated across multiple forecasting horizons and repeated random seeds.

## Highlights

- Multivariate time-series forecasting with chronological train/validation/test splits.
- Proposed **TSF-Resonator** model combining multi-scale temporal trend extraction and low-rank spectral filtering.
- Baselines included:
  - Persistence
  - DLinear
  - SparseTSF
  - MixLinear
  - PatchTST
  - iTransformer
- Multi-horizon evaluation, e.g. 12, 24, 48, and 96 steps.
- Multi-seed repeated experiments with mean, standard deviation, confidence interval summaries, and paired statistical tests.
- Runtime and efficiency benchmarking, including inference latency, throughput, parameter count, training time, and peak GPU memory.

## Repository Structure

```text
.
├── config/
│   ├── __init__.py
│   └── global_config_variables.py
├── models/
│   ├── __init__.py
│   ├── dlinear.py
│   ├── itransformer.py
│   ├── mixlinear.py
│   ├── patchtst.py
│   ├── sparsetsf.py
│   └── tsf_resonator.py
├── src/
│   ├── data_process_variables.py
│   ├── data_process_n2o_2min_all_variables.py
│   ├── layers.py
│   └── utils_variables.py
├── run_overall_update_all_variables.py
├── run_n2o_external_generalization_2min_all_variables.py
├── requirements.txt
└── README.md
```

> Note: Do not commit `__pycache__/` folders or local output folders to GitHub. Add them to `.gitignore` before pushing the repository.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you need a specific CUDA-enabled PyTorch build, install PyTorch from the official PyTorch selector first, then install the remaining packages from `requirements.txt`.

## Data Preparation

For the main all-variable experiment, the default configuration expects a CSV file at:

```text
data/all_data_water_yh.csv
```

The default time column is:

```text
_time
```

You can modify these settings in:

```text
config/global_config_variables.py
```

Important configuration fields include:

```python
self.data_path = os.path.join(self.base_dir, "data", "all_data_water_yh.csv")
self.output_dir = os.path.join(self.base_dir, "outputs_all_variables")
self.time_col = "_time"
self.exclude_cols = []
self.seq_len = 96
self.pred_len = 96
self.train_ratio = 0.7
self.val_ratio = 0.1
self.test_ratio = 0.2
```

The main data processor uses all original numeric variables, excluding the time column and any columns listed in `exclude_cols`. The target column is selected through the experiment script by using the `--targets` argument.

## Main Experiment

Run the main multi-target, multi-horizon experiment:

```bash
python run_overall_update_all_variables.py \
  --targets "out TP" "out orthophosphate" "out TN" \
  --models Persistence DLinear SparseTSF MixLinear PatchTST iTransformer TSF-Resonator \
  --pred_lens 12 24 48 96 \
  --num_seeds 5 \
  --epochs 80 \
  --batch_size 32
```

A smaller sanity-check run can be launched with:

```bash
python run_overall_update_all_variables.py \
  --targets "out TP" \
  --models Persistence DLinear TSF-Resonator \
  --pred_lens 12 \
  --num_seeds 1 \
  --epochs 1 \
  --patience 1
```

Useful command-line arguments:

| Argument | Description | Default |
|---|---|---|
| `--targets` | Target columns to evaluate | `"out TP" "out orthophosphate" "out TN"` |
| `--models` | Models to train/evaluate | Persistence, DLinear, SparseTSF, MixLinear, PatchTST, iTransformer, TSF-Resonator |
| `--pred_lens` | Forecasting horizons | `12 24 48 96` |
| `--num_seeds` | Number of random seeds | `5` |
| `--seeds` | Explicit comma-separated seed list | `None` |
| `--epochs` | Maximum training epochs | `80` |
| `--patience` | Early stopping patience | `10` |
| `--batch_size` | Training batch size | `32` |
| `--learning_rate` | Learning rate | `5e-4` |
| `--weight_decay` | Weight decay | `1e-4` |
| `--runtime_repeats` | Number of runtime benchmark repetitions | `5` |
| `--runtime_batch_size` | Batch size for runtime benchmarking | `64` |
| `--deterministic` | Enable deterministic settings where possible | disabled |

## N2O 2-Min External Generalization Experiment

The repository also includes an external generalization script for 2-minute N2O forecasting:

```bash
python run_n2o_external_generalization_2min_all_variables.py \
  --data_path data/n2o_2min_half_year.csv \
  --output_dir outputs/n2o_2min_external_generalization \
  --targets "BIOLOGY.LINE 3 TANK 1.N2O value" \
  --models Persistence SeasonalNaive TSF-Resonator MixLinear DLinear SparseTSF PatchTST iTransformer \
  --seq_len 96 \
  --pred_lens 12 24 48 96 \
  --sample_minutes 2 \
  --num_seeds 5
```

For a quick logic check:

```bash
python run_n2o_external_generalization_2min_all_variables.py \
  --data_path data/n2o_2min_half_year.csv \
  --dry_run
```

### Import compatibility note

If your final repository keeps the current file names:

```text
config/global_config_variables.py
src/data_process_n2o_2min_all_variables.py
```

make sure the N2O script imports match those names. For example:

```python
from config.global_config_variables import GlobalConfig
from src.data_process_n2o_2min_all_variables import (
    N2O2MinDataProcessor,
    TimeSeriesDataset,
    DEFAULT_N2O_TARGET,
)
```

Alternatively, you can create alias files named `config/global_config.py` and `src/data_process_n2o_2min.py`.

## Models

### TSF-Resonator

`TSF-Resonator` is a lightweight time-frequency forecasting model. It contains:

- Reversible instance normalization.
- Adaptive multi-scale temporal trend extraction.
- Seasonal and trend feature branches.
- Low-rank spectral filtering in the frequency domain.
- Learnable fusion between temporal and spectral representations.
- A linear projection head for forecasting.

### Baseline Models

The repository also implements or integrates:

- **Persistence**: last-observation-carried-forward baseline.
- **DLinear**: decomposition-linear model with trend and seasonal branches.
- **SparseTSF**: lightweight sparse forecasting baseline with RevIN.
- **MixLinear**: mixed temporal and frequency linear model.
- **PatchTST**: patch-based Transformer model for time-series forecasting.
- **iTransformer**: inverted Transformer-style multivariate forecaster.

## Outputs

By default, results are written to:

```text
outputs_all_variables/
```

The output directory contains:

```text
outputs_all_variables/
├── checkpoints/
├── figures/
├── logs/
└── predictions/
```

Typical output files include:

- Overall metrics by seed.
- Step-wise metrics by forecasting horizon.
- Runtime efficiency summaries.
- Mean/std/95% CI metric tables.
- LaTeX-ready metric tables.
- Pairwise statistical tests against TSF-Resonator.
- Prediction files saved as compressed `.npz` files.
- Figures in PNG, PDF, and SVG formats.

## Evaluation Metrics

The project reports:

- MSE
- MAE
- RMSE
- R²
- Step-wise metrics for each forecasting step
- Runtime metrics:
  - Total parameters
  - Trainable parameters
  - Training time
  - Average epoch time
  - Inference latency
  - Inference throughput
  - Peak GPU memory, when CUDA is available

For N2O experiments, high-event metrics are also computed:

- High-event MSE, MAE, RMSE, and R²
- Event precision
- Event recall
- Event F1 score

## Reproducibility

The experiment scripts support repeated runs through:

```bash
--num_seeds 5
```

or an explicit seed list:

```bash
--seeds 2021,2022,2023,2024,2025
```

Use deterministic settings where possible:

```bash
--deterministic
```

Note that exact reproducibility can still depend on the PyTorch, CUDA, cuDNN, GPU, and operating system versions.

## Recommended `.gitignore`

Before uploading to GitHub, consider adding a `.gitignore` file such as:

```gitignore
__pycache__/
*.py[cod]
.venv/
.env
.DS_Store

data/
outputs/
outputs_all_variables/
checkpoints/
figures/
logs/
predictions/

*.pth
*.pt
*.npz
*.csv
```

If you want to share example data or small processed samples, remove the relevant data patterns from `.gitignore` or place sample files in a dedicated `examples/` directory.

## License

Add a license file before public release if you want others to reuse the code. Common choices for research code include MIT, Apache-2.0, and BSD-3-Clause.

## Citation

If this repository supports a paper or thesis, replace the placeholder below with the final citation:

```bibtex
@misc{tsf_resonator,
  title  = {TSF-Resonator: Time-Frequency Resonance Network for Wastewater Time-Series Forecasting},
  author = {Your Name},
  year   = {2026},
  note   = {GitHub repository}
}
```
