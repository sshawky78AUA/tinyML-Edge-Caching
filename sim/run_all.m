function run_all(varargin)
% RUN_ALL  Reproduce the results of the paper, from the traces to the tables.
% Run it from the sim/ folder in MATLAB or GNU Octave. With no arguments it
% runs every step below except 'train' (the trained models are included in
% models/, so training is optional). Results are written to sim/results/,
% replacing the files of the paper; use git to compare or restore them.
%
% Steps, in the order they run:
%  'traces'    build data/trace_s1p.mat and data/trace_mt.mat
%  'export'    write the training examples to data/train_syn and data/train_mt
%  'train'     train and quantize every model (calls Python and TensorFlow)
%  'baselines' select the baseline b2 and compute Belady's reference
%  'validate'  check the simulator against Che's approximation and the
%              Pollaczek-Khinchine mean wait (Section VI-E)
%  'benchmark' benchmark protocol on the synthetic trace (Sections VI-F, VII)
%  'mt'        benchmark on MovieTweetings (Section VII-F)
%  'sweeps'    rate sweep, zero-latency check and cache-size sweep (RQ4, RQ5)
%  'aggregate' collect results/res2*.mat into results/results2*.json
%  'report'    statistics, tables and figures (calls Python: analyze2.py,
%              fill2.py); writes the tables into the paper source
%
% Options (name, value):
%  'steps'     cell array of step names (default: all but 'train')
%  'platforms' platform classes for 'benchmark' and 'sweeps' (default 1:4)
%  'seeds'     training seeds for 'benchmark' and 'aggregate' (default 1:5)
%  'mtplatforms', 'mtseeds'  for 'mt' (default [1 3] and 1:3)
%  'python'    Python command (default 'python3')
%
% Examples:
%  run_all                                         % everything except training
%  run_all('steps', {'benchmark'}, 'platforms', 3, 'seeds', 1)
%  run_all('steps', {'aggregate', 'report'})       % tables and figures only
%
% Runtime on one CPU core (GNU Octave 8): 'benchmark' takes about 15 to 40
% minutes per platform and seed, so the full synthetic benchmark (4 x 5)
% takes several hours; 'sweeps' about 2 hours; the other steps minutes.
% Edit params_default.m to change the parameters of the study.

o = struct('steps', {{'traces','export','baselines','validate','benchmark','mt','sweeps','aggregate','report'}}, ...
           'platforms', 1:4, 'seeds', 1:5, 'mtplatforms', [1 3], 'mtseeds', 1:3, 'python', 'python3');
for a = 1:2:numel(varargin), o.(varargin{a}) = varargin{a+1}; end
if ischar(o.steps), o.steps = {o.steps}; end
if ~exist('results', 'dir'), mkdir('results'); end
has = @(s) any(strcmp(o.steps, s));
t00 = tic;

if has('traces'),   step('traces');   build_traces(); end
if has('export'),   step('export');   export_all(); end
if has('train')
  step('train');
  for s = o.seeds
    sh(o, sprintf('cd ../models && DATA=../data/train_syn %s train.py %d', o.python, s));
  end
  sh(o, sprintf('cd ../models && %s select_theta3.py ../data/train_syn weights', o.python));
  sh(o, sprintf('cd ../models && %s equivalence2.py ../data/train_syn . 0.15 1234 %s', o.python, seedlist(o.seeds)));
  for s = o.mtseeds
    sh(o, sprintf('cd ../models/mt && DATA=../../data/train_mt STAGES=123 %s ../train.py %d', o.python, s));
  end
  sh(o, sprintf('cd ../models && %s select_theta3.py ../data/train_mt mt/weights', o.python));
  sh(o, sprintf('cd ../models && %s equivalence2.py ../data/train_mt mt 0.10 123 %s', o.python, seedlist(o.mtseeds)));
end
if has('baselines'), step('baselines'); baselines(); end
if has('validate')
  step('validate'); V = validate_sim(params_default()); save('results/results_validation.mat', 'V', '-v7');
end
if has('benchmark')
  for ip = o.platforms, step(sprintf('benchmark, Class %d', ip)); exp_platform2(ip, o.seeds); end
end
if has('mt')
  for ip = o.mtplatforms, step(sprintf('MovieTweetings, Class %d', ip)); exp_platform2(ip, o.mtseeds, 'mt'); end
end
if has('sweeps')
  for ip = o.platforms
    step(sprintf('rate sweep, Class %d', ip)); exp_rate2(ip, [1/16 1/4 1 4 16 64 256], 1);
    if ip > 1, exp_ratezl(ip, 256, 1); end
    if ip <= 3, step(sprintf('cache sweep, Class %d', ip)); exp_cache2(ip, o.seeds, [0.005 0.01 0.02 0.05]); end
  end
end
if has('aggregate'), step('aggregate'); aggregate2('', o.seeds); aggregate2('mt', o.mtseeds); end
if has('report')
  step('report');
  sh(o, sprintf('%s analyze2.py', o.python)); sh(o, sprintf('%s analyze2.py mt', o.python));
  sh(o, sprintf('%s fill2.py', o.python));
end
fprintf('\nDone in %.1f minutes.\n', toc(t00)/60);
end

function step(name)
fprintf('\n==== %s (%s) ====\n', name, datestr(now, 'HH:MM:SS'));
end

function sh(o, cmd)
fprintf('$ %s\n', cmd);
st = system(cmd);
if st ~= 0, error('run_all:shell', 'Command failed (%d): %s', st, cmd); end
end

function s = seedlist(seeds)
s = strjoin(arrayfun(@(x) sprintf('%d', x), seeds, 'UniformOutput', false), ',');
end
