function select_b2(tag)
% SELECT_B2  Baseline b_2 of a trace: the best of LRU, LFU and ARC by hit
% ratio on the validation part (instant decisions, Class 3 profile).
P = params_default(); tf = 'trace_s1p.mat'; out = 'results_b2.mat';
if strcmp(tag, 'mt'), tf = 'trace_mt.mat'; out = 'results_b2_mt.mat'; P.modeldir = '../models_mt/'; end
load(tf); segV = [tr.Ttr - 4*3600, tr.Ttr, tr.Tva];
kt = find(tr.t >= tr.Tva); C = P.Cfrac*sum(tr.s(unique(tr.f(kt))));
[WM, pfs] = load_all(P, 1); pf = pfs{3}; Hs = struct(); best = 0;
for b = {'lru', 'lfu', 'arc'}
  R = run_sim(tr, P, struct('impl', {{'b','b','b','b'}}, 's2base', b{1}, 'C', C), pf, WM, segV, P.Nh_inf, struct('zerolat', 1));
  Hs.(b{1}) = R.H; printf('%s H_va = %.4f\n', b{1}, R.H);
  if R.H > best, best = R.H; b2 = b{1}; end
end
save('-v7', out, 'b2', 'Hs');
end
