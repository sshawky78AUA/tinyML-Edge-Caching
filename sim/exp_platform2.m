function exp_platform2(ip, seeds, tag)
% EXP_PLATFORM2  Benchmark protocol on platform class ip (Section VI-F):
%  1. baseline and every feasible single-stage configuration (val + test);
%  2. energy-neutral selection (31) and loss split (30) of the selected models;
%  3. energy-neutral combined pipeline (33) with eta = 0;
%  4. hit-ratio-oriented pipeline: S2 = model with the largest validation
%     gain; the prefetching stages are measured behind it (reference pi^A)
%     and the pipeline is searched without the energy constraint.
if nargin < 3, tag = ''; end
P = params_default(); tf = 'trace_s1p.mat'; bf = 'results_b2.mat'; stages = 1:4;
if strcmp(tag, 'mt'), tf = 'trace_mt.mat'; bf = 'results_b2_mt.mat'; P.modeldir = '../models_mt/'; stages = 1:3; P.theta3 = 0.10; end   % theta3 by F1 on the MT validation part
load(tf);
segV = [tr.Ttr - 4*3600, tr.Ttr, tr.Tva]; segT = [tr.Tva - 4*3600, tr.Tva, tr.Tend];
kt = find(tr.t >= tr.Tva); C = P.Cfrac*sum(tr.s(unique(tr.f(kt)))); smean = mean(tr.s(tr.f));
S = load(bf); b2 = S.b2;
stg = {{'s1_lin','s1_mlp_s','s1_mlp_l'},{'s2_lin','s2_mlp_s','s2_mlp_l'}, ...
       {'s3_lin','s3_mlp_s','s3_mlp_l'},{'s4_lin','s4_mlp_s','s4_mlp_l'}};
for seed = seeds
  [WM, pfs] = load_all(P, seed); pf = pfs{ip};
  base = {'b','b','b','b'};
  run = @(impl, seg, Nh, zl) run_sim(tr, P, struct('impl', {impl}, 's2base', b2, 'C', C), pf, WM, seg, Nh, struct('zerolat', zl));
  Nh0 = nh_for(base, pf, P, C, smean);
  O = struct(); O.C = C; O.b2 = b2; O.Nh0 = Nh0; O.pf = pf.name; O.theta3 = P.theta3;
  O.base_va = run(base, segV, Nh0, 0); O.base_te = run(base, segT, Nh0, 0);
  % ---- 1-2: single stages, energy-neutral selection ----
  best = base; m2H = 'b'; bestH2 = 0;
  for j = stages
    bestgain = 0;
    for a = 1:3
      m = stg{j}{a}; impl = base; impl{j} = m;
      [Nh, feas] = nh_for(impl, pf, P, C, smean);
      O.single.(m).feas = feas; O.single.(m).Nh = Nh;
      if ~feas, continue; end
      Rv = run(impl, segV, Nh, 0); O.single.(m).va = Rv;
      dH = Rv.H - O.base_va.H; G = O.base_va.Esys - Rv.Esys;
      if dH > bestgain && G >= 0, bestgain = dH; best{j} = m; end
      if j == 2 && dH > bestH2 && Rv.rho < 1, bestH2 = dH; m2H = m; end
      printf('C%d s%d %s dHva=%.4f Gva=%.2f Nh=%d\n', ip, seed, m, dH, G, Nh); fflush(stdout);
    end
    % the test part evaluates the selected models only (S2: both selections)
    for m = unique([best(j), {m2H}(j == 2)])
      if strcmp(m{1}, 'b'), continue; end
      impl = base; impl{j} = m{1};
      O.single.(m{1}).te = run(impl, segT, O.single.(m{1}).Nh, 0);
      printf('C%d s%d %s dHte=%.4f\n', ip, seed, m{1}, O.single.(m{1}).te.H - O.base_te.H); fflush(stdout);
    end
  end
  O.best = best;
  % loss split (30): each configuration keeps its own history length on p_bar
  O.split = struct();
  bar_b = run(base, segT, Nh0, 1); inf_b = run(base, segT, P.Nh_inf, 1);
  for j = 1:4
    m = best{j}; if strcmp(m, 'b'), continue; end
    impl = base; impl{j} = m; Nh = nh_for(impl, pf, P, C, smean);
    bar_m = run(impl, segT, Nh, 1); inf_m = run(impl, segT, P.Nh_inf, 1);
    dHp = O.single.(m).te.H - O.base_te.H; dHbar = bar_m.H - bar_b.H; dHinf = inf_m.H - inf_b.H;
    O.split.(m) = [dHinf, dHbar, dHp, dHinf - dHbar, dHbar - dHp];
  end
  % ---- 3: energy-neutral combined pipeline ----
  [O.comb, O.comb_va] = search(best, base, O.base_va, true);
  O.comb_Nh = nh_for(O.comb, pf, P, C, smean); O.comb_te = run(O.comb, segT, O.comb_Nh, 0);
  sumdH = 0;
  for j = 1:4, if ~strcmp(O.comb{j}, 'b'), sumdH = sumdH + O.single.(O.comb{j}).te.H - O.base_te.H; end, end
  O.I = (O.comb_te.H - O.base_te.H) - sumdH;
  % ---- 4: hit-ratio-oriented pipeline ----
  A = base; A{2} = m2H; O.m2H = m2H; NhA = nh_for(A, pf, P, C, smean);
  O.A_va = run(A, segV, NhA, 0); O.A_te = run(A, segT, NhA, 0);
  bestA = A;
  for j = intersect([1 3 4], stages)
    bg = 0;
    for a = 1:3
      m = stg{j}{a}; impl = A; impl{j} = m; [Nh, feas] = nh_for(impl, pf, P, C, smean);
      O.behind.(m).feas = feas; if ~feas, continue; end
      Rv = run(impl, segV, Nh, 0); O.behind.(m).va = Rv;
      dH = Rv.H - O.A_va.H;
      if dH > bg && Rv.rho < 1, bg = dH; bestA{j} = m; end
      printf('C%d s%d behind %s dHva=%.4f\n', ip, seed, m, dH); fflush(stdout);
    end
    if ~strcmp(bestA{j}, 'b')                       % test run of the selected model only
      m = bestA{j}; impl = A; impl{j} = m;
      O.behind.(m).te = run(impl, segT, nh_for(impl, pf, P, C, smean), 0);
    end
  end
  O.bestA = bestA;
  [O.combH, O.combH_va] = search(bestA, A, O.A_va, false);   % S2 kept at m2H
  O.combH_Nh = nh_for(O.combH, pf, P, C, smean); O.combH_te = run(O.combH, segT, O.combH_Nh, 0);
  printf('C%d s%d comb0 %s dHte=%.4f | combH %s dHte=%.4f dEsys=%.1f\n', ip, seed, strjoin(O.comb, ','), ...
    O.comb_te.H - O.base_te.H, strjoin(O.combH, ','), O.combH_te.H - O.base_te.H, O.combH_te.Esys - O.base_te.Esys); fflush(stdout);
  save('-v7', sprintf('res2%s_C%d_seed%d.mat', tag, ip, seed), 'O');
end

  function [bestc, bestR] = search(cand, ref, Rref, energy)
    % exhaustive search over the stages where cand differs from ref
    act = find(~strcmp(cand, ref)); nb = numel(act);
    bestc = ref; bestR = Rref; bestHv = Rref.H;
    if energy && Rref.Esys > O.base_va.Esys, bestHv = -inf; end
    for mask = 1:(2^nb - 1)
      impl = ref; for i = 1:nb, if bitget(mask, i), impl{act(i)} = cand{act(i)}; end, end
      [Nh, feas] = nh_for(impl, pf, P, C, smean); if ~feas, continue; end
      Rv = run(impl, segV, Nh, 0);
      okE = ~energy || Rv.Esys <= O.base_va.Esys;
      if Rv.rho < 1 && okE && Rv.H > bestHv, bestHv = Rv.H; bestc = impl; bestR = Rv; end
    end
  end
end
