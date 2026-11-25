function exp_cache2(ip, seeds, fracs, tag)
% EXP_CACHE2  Cache-size sweep (RQ5): the baseline and the two pipelines
% selected at the nominal cache size are evaluated on the test part with
% capacity C = frac * (unique test bytes).
if nargin < 4, tag = ''; end
P = params_default(); tf = '../data/trace_s1p.mat'; bf = 'results/results_b2.mat';
if strcmp(tag, 'mt'), tf = '../data/trace_mt.mat'; bf = 'results/results_b2_mt.mat'; P.modeldir = '../models/mt/'; P.theta3 = 0.10; end
S_ = load(tf); tr = S_.tr; S = load(bf); b2 = S.b2;
segT = [tr.Tva - 4*3600, tr.Tva, tr.Tend];
kt = find(tr.t >= tr.Tva); U0 = sum(tr.s(unique(tr.f(kt)))); smean = mean(tr.s(tr.f));
out = struct('fracs', fracs, 'seeds', seeds, 'H', zeros(numel(seeds), numel(fracs), 3), 'Esys', zeros(numel(seeds), numel(fracs), 3), 'D', zeros(numel(seeds), numel(fracs), 3));
for a = 1:numel(seeds)
  s = seeds(a); [WM, pfs] = load_all(P, s); pf = pfs{ip};
  Q = load(sprintf('results/res2%s_C%d_seed%d.mat', tag, ip, s)); O = Q.O;
  cf = {{'b','b','b','b'}, O.comb, O.combH};
  for b = 1:numel(fracs)
    C = fracs(b)*U0;
    for c = 1:3
      [Nh, feas] = nh_for(cf{c}, pf, P, C, smean);
      if ~feas, out.H(a,b,c) = NaN; continue; end
      R = run_sim(tr, P, struct('impl', {cf{c}}, 's2base', b2, 'C', C), pf, WM, segT, Nh, struct());
      out.H(a,b,c) = R.H; out.Esys(a,b,c) = R.Esys; out.D(a,b,c) = R.D;
    end
    fprintf('C%d s%d frac=%.3f H=%s\n', ip, s, fracs(b), mat2str(squeeze(out.H(a,b,:))', 4));
  end
end
save(sprintf('results/cache2%s_C%d.mat', tag, ip), 'out', '-v7');
end
