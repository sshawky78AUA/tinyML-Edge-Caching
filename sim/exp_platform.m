function exp_platform(ip, seeds)
% EXP_PLATFORM  Benchmark protocol of Section III-H on platform class ip:
% stage gains, model selection on validation, loss split, combined pipeline.
P = params_default(); load('trace_s1p.mat');
segV = [tr.Ttr - 4*3600, tr.Ttr, tr.Tva]; segT = [tr.Tva - 4*3600, tr.Tva, tr.Tend];
kt = find(tr.t >= tr.Tva); C = P.Cfrac*sum(tr.s(unique(tr.f(kt))));
smean = mean(tr.s(tr.f));
S = load('results_b2.mat'); b2 = S.b2;
stg = {{'s1_lin','s1_mlp_s','s1_mlp_l'},{'s2_lin','s2_mlp_s','s2_mlp_l'}, ...
       {'s3_lin','s3_mlp_s','s3_mlp_l'},{'s4_lin','s4_mlp_s','s4_mlp_l'}};
for seed = seeds
  [WM, pfs] = load_all(P, seed); pf = pfs{ip};
  base = {'b','b','b','b'};
  run = @(impl, seg, Nh, zl) run_sim(tr, P, struct('impl', {impl}, 's2base', b2, 'C', C), pf, WM, seg, Nh, struct('zerolat', zl));
  Nh0 = nh_for(base, pf, P, C, smean);
  O = struct(); O.C = C; O.b2 = b2; O.Nh0 = Nh0; O.pf = pf.name;
  O.base_va = run(base, segV, Nh0, 0); O.base_te = run(base, segT, Nh0, 0);
  best = base;
  for j = 1:4
    bestgain = 0;
    for a = 1:3
      m = stg{j}{a}; impl = base; impl{j} = m;
      [Nh, feas] = nh_for(impl, pf, P, C, smean);
      key = sprintf('%s', m);
      O.single.(key).feas = feas; O.single.(key).Nh = Nh;
      if ~feas, continue; end
      Rv = run(impl, segV, Nh, 0); Rt = run(impl, segT, Nh, 0);
      O.single.(key).va = Rv; O.single.(key).te = Rt;
      dH = Rv.H - O.base_va.H; G = O.base_va.Esys - Rv.Esys;
      if dH > bestgain && G >= 0, bestgain = dH; best{j} = m; end
      printf('C%d seed%d %s dHva=%.4f Gva=%.2f dHte=%.4f Nh=%d\n', ip, seed, m, dH, G, Rt.H - O.base_te.H, Nh); fflush(stdout);
    end
  end
  O.best = best;
  % ---- loss split (eq. 38) for the selected models ----
  O.split = struct();
  for j = 1:4
    if strcmp(best{j}, 'b'), continue; end
    impl = base; impl{j} = best{j}; Nh = nh_for(impl, pf, P, C, smean);
    bar_m = run(impl, segT, Nh, 1);   bar_b = run(base, segT, Nh, 1);
    inf_m = run(impl, segT, P.Nh_inf, 1); inf_b = run(base, segT, P.Nh_inf, 1);
    dHp = O.single.(best{j}).te.H - O.base_te.H;
    dHbar = bar_m.H - bar_b.H; dHinf = inf_m.H - inf_b.H;
    O.split.(best{j}) = [dHinf, dHbar, dHp, dHinf - dHbar, dHbar - dHp];
  end
  % ---- combined pipeline (eq. 31), selected on validation ----
  act = find(~strcmp(best, 'b')); nb = numel(act); cand = {};
  bestH = O.base_va.H; O.comb = base; O.comb_va = O.base_va;
  for mask = 1:(2^nb - 1)
    impl = base; for i = 1:nb, if bitget(mask, i), impl{act(i)} = best{act(i)}; end, end
    [Nh, feas] = nh_for(impl, pf, P, C, smean); if ~feas, continue; end
    Rv = run(impl, segV, Nh, 0);
    if Rv.rho < 1 && Rv.Esys <= O.base_va.Esys && Rv.H > bestH
      bestH = Rv.H; O.comb = impl; O.comb_va = Rv;
    end
  end
  [Nh, ~] = nh_for(O.comb, pf, P, C, smean); O.comb_Nh = Nh;
  O.comb_te = run(O.comb, segT, Nh, 0);
  sumdH = 0;
  for j = 1:4, if ~strcmp(O.comb{j}, 'b'), sumdH = sumdH + O.single.(O.comb{j}).te.H - O.base_te.H; end, end
  O.I = (O.comb_te.H - O.base_te.H) - sumdH;
  printf('C%d seed%d combined=%s dH=%.4f I=%.4f\n', ip, seed, strjoin(O.comb, ','), O.comb_te.H - O.base_te.H, O.I);
  save('-v7', sprintf('results_C%d_seed%d.mat', ip, seed), 'O');
end
end
