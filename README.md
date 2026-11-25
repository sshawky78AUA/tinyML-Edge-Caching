# TinyML for Edge Caching: How the Hardware Platform Shapes the Gain

Code, data and trained models for the paper by Sameh Hosny and John Tadrous. The simulator models an edge cache node whose caching decisions are made by a pipeline of four stages (S1 popularity prediction, S2 admission and eviction, S3 next-request prediction, S4 next-cell prediction). Each stage runs a non-learned baseline or a tinyML model on one of four microcontroller platforms, and the simulator reports the hit ratio, delay, backhaul traffic and energy of the node and of the system.

The simulations run in **MATLAB** or **GNU Octave** with no toolboxes. Python is needed only to retrain the models and to write the tables and figures.

## Quick start: one simulation with your own parameters

```matlab
cd sim
run_demo                                       % Class 3 platform, S2 MLP-S model, synthetic trace
run_demo('platform', 1)                        % the same pipeline on the Cortex-M3
run_demo('pipeline', {'b','s2_mlp_l','s3_mlp_s','b'}, 'cache', 0.005)
run_demo('pipeline', 'bestH', 'eb', 0.1e-6)    % hit-ratio-oriented pipeline, cheaper backhaul
run_demo('rate', 64, 'platform', 2)            % 64 times the request rate
run_demo('trace', 'mt')                        % MovieTweetings trace
R = run_demo('W2', 1800, 'theta2', 0.4);       % any field of params_default.m
```

Each call simulates the baseline pipeline and the chosen pipeline on the test part of the trace (about 30 s each) and prints a comparison:

```
Class 3 (Alif E3, Cortex-M55 + Ethos-U55), syn trace, seed 1, cache 1.00% of unique test bytes, rate x1
pipeline: b, s2_mlp_s, b, b (history 20000 requests); baseline: b2 = arc (history 20000)

                               baseline     pipeline       change
Hit ratio H                      0.1951       0.2069      +0.0118
Byte hit ratio                   0.1748       0.1872      +0.0124
Mean delay (ms)                  173.77       171.62        -2.15
Backhaul traffic (MB)             728.4        717.8        -10.5
Node energy E_EN (J)               4.38         4.40        +0.01
System energy E_sys (J)           819.2        808.7        -10.5
Hit-ratio gain: +1.18 points (+6.1%); system energy saved: +10.5 J
```

| Option | Meaning | Default |
|---|---|---|
| `trace` | `'syn'` (synthetic) or `'mt'` (MovieTweetings) | `'syn'` |
| `platform` | 1 STM32F103 (Cortex-M3), 2 STM32F407 (Cortex-M4F), 3 Alif E3 (Cortex-M55 + Ethos-U55), 4 GAP8 | 3 |
| `seed` | training seed of the models (1 to 5; 1 to 3 for `mt`) | 1 |
| `pipeline` | one entry per stage S1 to S4: `'b'`, `'s<j>_lin'`, `'s<j>_mlp_s'` or `'s<j>_mlp_l'`; or `'best'` / `'bestH'` for the energy-neutral / hit-ratio-oriented pipeline the benchmark selected | `{'b','s2_mlp_s','b','b'}` |
| `cache` | cache capacity as a share of the unique test bytes | 0.01 |
| `rate` | request-rate multiplier | 1 |
| `eb`, `ea` | backhaul and access energy per byte (J/B) | 1e-6, 1e-7 |
| `Rb`, `db`, `Ra`, `da` | link rates (B/s) and propagation delays (s) | see `params_default.m` |
| `theta2`, `theta3`, `L`, `W2`, ... | any field of `params_default.m` | |
| `zerolat` | `true` removes the decision latency of the platform | `false` |
| `nh` | request-history length (default: the longest that fits in RAM next to the models) | |

The platform profiles (clock, memory, energy per cycle, sleep and wake-up costs) are in `sim/platforms.m`, and the cycle counts of the models in `fw/cycles.json` and `models/vela_summary.json`. Edit them to study another platform.

## Reproducing the paper

From `sim/`, in MATLAB or Octave:

```matlab
run_all                                          % every step except training
run_all('steps', {'benchmark'}, 'platforms', 3, 'seeds', 1)
run_all('steps', {'aggregate', 'report'})        % tables and figures only
```

| Step | What it does | Main script | Time (one core) |
|---|---|---|---|
| `traces` | builds `data/trace_s1p.mat` and `data/trace_mt.mat` | `build_traces.m` | 1 min |
| `export` | writes the training examples to `data/train_syn/` and `data/train_mt/` | `export_all.m` | 2 min |
| `train` | trains and quantizes every model (Python, TensorFlow) | `models/train.py` | a few minutes per seed |
| `baselines` | baseline b2 (best of LRU, LFU, ARC) and Belady's reference | `baselines.m` | 1 min |
| `validate` | simulator checks against Che's approximation and the Pollaczek-Khinchine formula | `validate_sim.m` | 30 s |
| `benchmark` | benchmark protocol on the synthetic trace, Classes 1 to 4, seeds 1 to 5 | `exp_platform2.m` | 15 to 30 min per class and seed |
| `mt` | benchmark on MovieTweetings, Classes 1 and 3, seeds 1 to 3 | `exp_platform2(ip, seeds, 'mt')` | 15 to 30 min per class and seed |
| `sweeps` | rate sweep, zero-latency check, cache-size sweep | `exp_rate2.m`, `exp_ratezl.m`, `exp_cache2.m` | 2 to 3 h |
| `aggregate` | collects `results/res2*.mat` into `results/results2*.json` | `aggregate2.m` | seconds |
| `report` | statistics, the tables written into the paper source, Figs. 2 to 4 | `analyze2.py`, `fill2.py` | seconds |

The results of the paper are included in `sim/results/`, so `run_all('steps', {'aggregate', 'report'})` rebuilds the tables and figures at once. New runs overwrite these files; use `git diff` or `git checkout` to compare or restore them.

To study another setting, edit `sim/params_default.m` (trace, links, cache, stage periods, horizons and thresholds) and run the steps again. The benchmark then selects the best model and platform for each stage under the new setting.

### Exactness

With GNU Octave 8.4, `build_traces`, `export_all` and `baselines` reproduce the files of the paper byte for byte, `exp_platform2` reproduces every value of the included result files, and the report step reproduces the tables and figures of the paper. The included benchmark files also hold test-part runs of models that the benchmark did not select, from an earlier run; the current script follows the protocol of the paper and runs the test part only for the selected models, and no table uses those extra runs.

Two differences are expected elsewhere:

- **MATLAB** uses a different random number generator for `rand('state', ...)`. The included traces, models and results are the same in both programs, but a synthetic trace rebuilt in MATLAB has the same statistics with different individual requests, and simulations give close but not identical numbers.
- **Training** with TensorFlow is not bit-reproducible across machines and versions. The included weights (`models/weights/`, `models/mt/weights/`) reproduce the paper; retrained models reach similar validation scores with different weights.

## Folders

- `sim/`: the simulator and the experiments (MATLAB/Octave), and the Python scripts for the tables and figures.
  - `run_sim.m`: the event-driven simulator (Section III); `platforms.m`: the platform profiles (Table IV); `params_default.m`: all parameters.
  - `feat_s1.m` to `feat_s4.m`: the features of each stage; `model_eval.m`: the model forward pass; `nh_for.m`: the memory check (Definition 1).
  - `gen_trace.m`, `prep_trace.m`, `build_mt.m`: the traces; `export_training.m`, `export_s3c.m`: the training examples.
  - `run_demo.m`: one simulation with your parameters; `run_all.m`: the whole study.
  - `results/`: the results of the paper.
- `models/`: `train.py` (training and int8 quantization), `equivalence2.py` (int8 against float check), `select_theta3.py`, the trained weights and per-seed summaries, and the Vela estimates for the Ethos-U55. The MovieTweetings models are in `models/mt/`.
- `fw/`: the int8 kernels (`kern.c`) and the QEMU-based cycle model (`cycles.py`, `cycles.json`) for the Cortex-M3 and Cortex-M4.
- `data/`: the two traces of the paper and the MovieTweetings 2017 window (`mt2017.csv`: time scaled by 158.4, user, item).
- `paper/`: the LaTeX source, the figures and `IEEEtran.cls`.

## Requirements

- MATLAB R2019b or later, or GNU Octave 6 or later.
- Python 3.10 or later with the packages in `requirements.txt` (`pip install -r requirements.txt`), for training, tables and figures. Tested with Python 3.13, TensorFlow 2.21, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11.
- The `train` and `report` steps call `python3` through the shell; on Windows, use `run_all(..., 'python', 'python')` or run the Python scripts by hand.

MovieTweetings: S. Dooms, T. De Pessemier and L. Martens, "MovieTweetings: a movie rating dataset collected from Twitter," CrowdRec Workshop at ACM RecSys 2013 (https://github.com/sidooms/MovieTweetings, MIT license).

## License and citation

The code is released under the MIT License (see `LICENSE`). If you use it, please cite the paper (see `CITATION.cff`). The MovieTweetings data keep the license of their source project.
