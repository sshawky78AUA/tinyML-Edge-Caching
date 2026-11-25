function baselines()
% BASELINES  Baseline b_2 of each trace (the best of LRU, LFU and ARC on the
% validation part) and the hit ratio of Belady's rule on the test part.
%  results/results_b2.mat, results/results_b2_mt.mat         : b2, Hs
%  results/results_belady.mat, results/results_belady_mt.mat : HB
if ~exist('results', 'dir'), mkdir('results'); end
select_b2('');
select_b2('mt');
P = params_default();
tags = {'', '_mt'}; files = {'../data/trace_s1p.mat', '../data/trace_mt.mat'};
for a = 1:2
  S_ = load(files{a}); tr = S_.tr;
  kt = find(tr.t >= tr.Tva); C = P.Cfrac*sum(tr.s(unique(tr.f(kt))));
  HB = belady(tr, C, [tr.Tva - 4*3600, tr.Tva, tr.Tend]);
  save(sprintf('results/results_belady%s.mat', tags{a}), 'HB', '-v7');
  fprintf('Belady%s: H = %.4f\n', tags{a}, HB);
end
end
