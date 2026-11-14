# TinyML for Edge Caching: How the Hardware Platform Shapes the Gain

## Folders

- `paper/`: the LaTeX source (`tinyML Edge Caching.tex`), the PDF, the four figure files and `IEEEtran.cls`. To build it, run `pdflatex` twice inside this folder.
- `sim/`: the MATLAB/GNU Octave simulator and experiment scripts, plus the Python scripts that make the tables and figures.
  - `run_sim.m`: the event-driven simulator (Section III).
  - `exp_platform2.m`: the benchmark protocol (Section VI-F). Use `exp_platform2(ip, seeds)` for the synthetic trace and `exp_mt(ip, seed)` for MovieTweetings.
  - `exp_rate2.m`, `exp_ratezl.m`: the rate sweep and its zero-latency check (RQ4).
  - `exp_cache2.m`: the cache-size sweep (RQ5).
  - `aggregate2.m` → `analyze2.py` → `fill2.py`: collect the results, compute the statistics, then write the tables into the paper and redraw the figures.
  - `gen_trace.m`, `prep_trace.m`, `build_mt.m`: build the synthetic trace and the MovieTweetings trace.
  - `export_training.m`, `export_s3c.m`: export the training examples.
  - `platforms.m`: the platform profiles (Table IV).
  - `results/`: the raw result files (`res2*_C*_seed*.mat`), the rate and cache sweeps, and the aggregated JSON files.
- `models/`: `train.py` (training and int8 quantization), `equivalence2.py` (int8-versus-float check), `select_theta3.py`, the per-seed summaries, the trained weights (`weights/`), and the Vela estimates. MovieTweetings models are in `models/mt/`.
- `fw/`: the int8 kernels (`kern.c`) and the QEMU-based cycle model (`cycles.py`, `cycles.json`) for the Cortex-M3 and Cortex-M4.
- `data/`: the synthetic trace (`trace_s1.mat`), and the MovieTweetings 2017 window (`mt2017.csv`: time scaled by 158.4, user, item) with the dataset's README.

## Reproducing the results

1. Training: `python3 models/train.py <seed>`. It needs TensorFlow; set `DATA=sim/data_mt STAGES=123` for MovieTweetings.
2. Benchmark: in Octave, from `sim/`, run `exp_platform2(ip, 1:5)` for the synthetic trace and `exp_mt(ip, seed)` for MovieTweetings.
3. Sweeps: `exp_rate2(ip, [1/16 1/4 1 4 16 64 256], 1)` and `exp_cache2(ip, 1:5, [0.005 0.01 0.02 0.05])`.
4. Tables and figures: run `aggregate2('', 1:5)`, then `python3 analyze2.py`, then `python3 fill2.py`.

MovieTweetings: S. Dooms, T. De Pessemier, L. Martens, CrowdRec at ACM RecSys 2013 (https://github.com/sidooms/MovieTweetings, MIT license).

## License and citation

The code is released under the MIT License (see `LICENSE`). If you use it, please cite the paper (see `CITATION.cff`). The MovieTweetings data keep the license of their source project.
