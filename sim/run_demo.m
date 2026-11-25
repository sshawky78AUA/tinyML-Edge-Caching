function out = run_demo(varargin)
% RUN_DEMO  Simulate the edge cache node with the parameters you choose and
% compare a pipeline of your choice with the baseline pipeline (every stage
% non-learned) on the test part of a trace. Prints the hit ratio, the byte
% hit ratio, the delay, the backhaul traffic, the energy of the node and of
% the system, and the load of the processor. One call takes a few minutes.
%
% Run it from the sim/ folder in MATLAB or GNU Octave:
%   run_demo                                    % nominal setting (Class 3, S2 MLP-S)
%   run_demo('platform', 1)                     % the same pipeline on the Cortex-M3
%   run_demo('pipeline', 'bestH', 'cache', 0.005)
%   run_demo('pipeline', {'b','s2_mlp_l','s3_mlp_s','b'}, 'eb', 0.1e-6)
%   run_demo('rate', 64, 'platform', 2)         % 64 times the request rate
%   R = run_demo('trace', 'mt');                % MovieTweetings
%
% Options (name, value pairs):
%  'trace'     'syn' (synthetic, default) or 'mt' (MovieTweetings)
%  'platform'  1 = STM32F103 (Cortex-M3), 2 = STM32F407 (Cortex-M4F),
%              3 = Alif E3 (Cortex-M55 + Ethos-U55, default), 4 = GAP8
%  'seed'      training seed of the models: 1..5 (synthetic), 1..3 (mt)
%  'pipeline'  1x4 cell, one entry per stage S1..S4 (popularity, admission
%              and eviction, next request, next cell): 'b' for the baseline
%              or a model 's<j>_lin', 's<j>_mlp_s', 's<j>_mlp_l'.
%              Default {'b','s2_mlp_s','b','b'}. Also 'best' (the
%              energy-neutral pipeline the benchmark selected for this
%              platform and seed) or 'bestH' (the hit-ratio-oriented one).
%              S4 needs user locations, so it is not available on 'mt'.
%  'cache'     cache capacity as a share of the unique test bytes (0.01)
%  'rate'      request-rate multiplier: the trace time is compressed by
%              this factor, as in the rate sweep of the paper (1)
%  'eb', 'ea'  backhaul and access energy per byte, J/B (1e-6, 1e-7)
%  'Rb', 'db'  backhaul rate (B/s) and propagation delay (s)
%  'Ra', 'da'  access rate (B/s) and propagation delay (s)
%  'theta2'    admission threshold of S2 (0.3)
%  'theta3', 'L'  prefetch threshold and maximum prefetches of S3
%  'zerolat'   true: decisions take no time (removes the platform latency)
%  'nh'        request-history length; default: the longest that fits in
%              the RAM of the platform next to the models (Definition 1)
% Any other field of params_default.m can be passed the same way, e.g.
% run_demo('W2', 1800) or run_demo('theta4', 0.3).
%
% The returned struct holds the full metrics of both runs (out.base,
% out.pipe) and the settings that were used.

% ---- options ----
o = struct('trace', 'syn', 'platform', 3, 'seed', 1, 'pipeline', {{'b','s2_mlp_s','b','b'}}, ...
           'cache', 0.01, 'rate', 1, 'zerolat', false, 'nh', []);
P = params_default(); pnames = fieldnames(P);
for a = 1:2:numel(varargin)
  k = varargin{a}; v = varargin{a+1};
  if isfield(o, k), o.(k) = v;
  elseif any(strcmp(pnames, k)), P.(k) = v;
  else, error('run_demo:option', 'Unknown option ''%s''.', k);
  end
end
ip = o.platform; seed = o.seed;
pnm = {'Class 1 (STM32F103, Cortex-M3)', 'Class 2 (STM32F407, Cortex-M4F)', ...
       'Class 3 (Alif E3, Cortex-M55 + Ethos-U55)', 'Class 4 (GAP8, RISC-V cluster)'};
if strcmp(o.trace, 'mt')
  tf = '../data/trace_mt.mat'; bf = 'results/results_b2_mt.mat'; tag = 'mt';
  P.modeldir = '../models/mt/'; if ~any(strcmp(varargin(1:2:end), 'theta3')), P.theta3 = 0.10; end
else
  tf = '../data/trace_s1p.mat'; bf = 'results/results_b2.mat'; tag = '';
end
S_ = load(tf); tr0 = S_.tr; S = load(bf); b2 = S.b2;
[WM, pfs] = load_all(P, seed); pf = pfs{ip};

% ---- pipeline ----
impl = o.pipeline;
if ischar(impl)
  fn = sprintf('results/res2%s_C%d_seed%d.mat', tag, ip, seed);
  if ~exist(fn, 'file'), error('run_demo:best', 'Run exp_platform2 for this platform and seed first (%s).', fn); end
  Q = load(fn);
  if strcmp(impl, 'best'), impl = Q.O.comb; else, impl = Q.O.combH; end
end
if strcmp(o.trace, 'mt') && ~strcmp(impl{4}, 'b')
  error('run_demo:s4', 'MovieTweetings has no user locations, so S4 must be ''b''.');
end
base = {'b','b','b','b'};

% ---- cache size and request rate ----
kt = find(tr0.t >= tr0.Tva); C = o.cache*sum(tr0.s(unique(tr0.f(kt)))); smean = mean(tr0.s(tr0.f));
g = o.rate; tr = tr0; Tva = tr0.Tva; Tend = tr0.Tend;
if g ~= 1          % compress time by g as in exp_rate2.m; the models keep seeing trace time
  t1 = tr0.t(1); tr.t = t1 + (tr0.t - t1)/g; tr.mobT = t1 + (tr0.mobT - t1)/g;
  Tva = t1 + (tr0.Tva - t1)/g; Tend = t1 + (tr0.Tend - t1)/g;
  P.T1 = P.T1/g; P.T4 = P.T4/g; P.W1 = P.W1/g; P.W2 = P.W2/g; P.W4 = P.W4/g;
  P.tscale = g; P.tref = t1;
end
seg = [Tva - 4*3600/g, Tva, Tend];   % 4-h warm-up, then the test part

% ---- history lengths (Definition 1) ----
[Nb, okb] = nh_for(base, pf, P, C, smean); [Np, okp] = nh_for(impl, pf, P, C, smean);
if ~okp, error('run_demo:memory', 'The pipeline does not fit in the memory of %s.', pnm{ip}); end
if ~isempty(o.nh), Nb = o.nh; Np = o.nh; end

% ---- simulate ----
fprintf('%s, %s trace, seed %d, cache %.2f%% of unique test bytes, rate x%g\n', ...
        pnm{ip}, o.trace, seed, 100*o.cache, g);
fprintf('pipeline: %s (history %d requests); baseline: b2 = %s (history %d)\n', strjoin(impl, ', '), Np, b2, Nb);
opt = struct('zerolat', o.zerolat); cfg = @(m) struct('impl', {m}, 's2base', b2, 'C', C);
tic; Rb = run_sim(tr, P, cfg(base), pf, WM, seg, Nb, opt);
Rp = run_sim(tr, P, cfg(impl), pf, WM, seg, Np, opt); el = toc;

% ---- report ----
rows = {'Hit ratio H', 'H', 1, '%.4f'; 'Byte hit ratio', 'HB', 1, '%.4f'; ...
        'Mean delay (ms)', 'D', 1e3, '%.2f'; 'Backhaul traffic (MB)', 'B', 1e-6, '%.1f'; ...
        'Prefetched (MB)', 'Spf', 1e-6, '%.1f'; 'Node energy E_EN (J)', 'EEN', 1, '%.2f'; ...
        'Link energy (J)', 'Elink', 1, '%.1f'; 'System energy E_sys (J)', 'Esys', 1, '%.1f'; ...
        'Processor load rho', 'rho', 1, '%.2g'};
fprintf('\n%-26s %12s %12s %12s\n', '', 'baseline', 'pipeline', 'change');
for a = 1:size(rows, 1)
  x = rows{a, 3}*Rb.(rows{a, 2}); y = rows{a, 3}*Rp.(rows{a, 2}); f = rows{a, 4};
  fprintf(['%-26s %12s %12s %12s\n'], rows{a, 1}, sprintf(f, x), sprintf(f, y), sprintf(['%+' f(2:end)], y - x));
end
fprintf('Hit-ratio gain: %+.2f points (%+.1f%%); system energy saved: %+.1f J\n', ...
        100*(Rp.H - Rb.H), 100*(Rp.H - Rb.H)/Rb.H, Rb.Esys - Rp.Esys);
fprintf('%d test requests, %.2f requests/s; %.0f s of simulation.\n', Rp.K, Rp.lambda, el);
out = struct('base', Rb, 'pipe', Rp, 'pipeline', {impl}, 'platform', pnm{ip}, 'C', C, 'P', P, 'options', o);
end
