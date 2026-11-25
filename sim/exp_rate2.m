function exp_rate2(ip, gammas, seed)
% EXP_RATE2  Rate sweep (RQ4 and the latency study): the trace is time-scaled
% by gamma (warm-up scaled by 1/gamma) and the baseline, the S2 MLP-S model,
% the selected S2 model, and the hit-ratio-oriented pipeline of the seed are
% evaluated on the test part.
P = params_default(); S_ = load('../data/trace_s1p.mat'); tr = S_.tr; tr0 = tr;
S = load('results/results_b2.mat'); b2 = S.b2;
kt = find(tr0.t >= tr0.Tva); C = P.Cfrac*sum(tr0.s(unique(tr0.f(kt)))); smean = mean(tr0.s(tr0.f));
[WM, pfs] = load_all(P, seed); pf = pfs{ip};
Q = load(sprintf('results/res2_C%d_seed%d.mat', ip, seed)); O = Q.O;
base = {'b','b','b','b'}; s2m = base; s2m{2} = 's2_mlp_s'; s2b = base; s2b{2} = O.best{2};
cfgs = {base, s2m, s2b, O.combH}; labels = {'base', 's2_mlp_s', 's2_best', 'combH'};
out = struct('gamma', gammas, 'labels', {labels}, 'cfgs', {cfgs}, 'runs', {{}});
for g = gammas
  tr = tr0; tr.t = tr0.t(1) + (tr0.t - tr0.t(1))/g; tr.mobT = tr0.t(1) + (tr0.mobT - tr0.t(1))/g;
  Tva = tr0.t(1) + (tr0.Tva - tr0.t(1))/g; Tend = tr0.t(1) + (tr0.Tend - tr0.t(1))/g;
  seg = [Tva - 4*3600/g, Tva, Tend]; P2 = P; P2.T1 = P.T1/g; P2.T4 = P.T4/g; P2.W1 = P.W1/g; P2.W2 = P.W2/g; P2.W4 = P.W4/g;
  P2.tscale = g; P2.tref = tr0.t(1);       % models see time in trace units
  rr = {};
  for a = 1:numel(cfgs)
    impl = cfgs{a}; [Nh, feas] = nh_for(impl, pf, P, C, smean);
    if ~feas, rr{a} = struct(); continue; end
    R = run_sim(tr, P2, struct('impl', {impl}, 's2base', b2, 'C', C), pf, WM, seg, Nh, struct());
    rr{a} = struct('H', R.H, 'HB', R.HB, 'Esys', R.Esys, 'EEN', R.EEN, 'Elink', R.Elink, 'rho', R.rho, ...
                   'drop', R.drop, 'nd', R.nd, 'D', R.D, 'lambda', R.lambda, 'Spf', R.Spf, 'bytesB', R.bytesB);
    fprintf('C%d g=%g %s H=%.4f rho=%.3g drop=%s\n', ip, g, labels{a}, R.H, R.rho, mat2str(R.drop));
  end
  out.runs{end+1} = rr;
end
save(sprintf('results/rate2_C%d.mat', ip), 'out', '-v7');
end
