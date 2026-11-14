function exp_split(ips, seeds)
% EXP_SPLIT  Loss split (30) for the selected models. Each of the three
% platforms p, p_bar, p_inf is simulated with the model and with the
% baseline; every configuration uses the history length that it has on that
% platform (p and p_bar: Definition 1 on p; p_inf: Nh_inf).
P = params_default(); load('trace_s1p.mat');
segT = [tr.Tva - 4*3600, tr.Tva, tr.Tend];
kt = find(tr.t >= tr.Tva); C = P.Cfrac*sum(tr.s(unique(tr.f(kt)))); smean = mean(tr.s(tr.f));
S = load('results_b2.mat'); b2 = S.b2; base = {'b','b','b','b'};
for ip = ips
  for seed = seeds
    fn = sprintf('results_C%d_seed%d.mat', ip, seed); Q = load(fn); O = Q.O;
    [WM, pfs] = load_all(P, seed); pf = pfs{ip};
    run = @(impl, Nh, zl) run_sim(tr, P, struct('impl', {impl}, 's2base', b2, 'C', C), pf, WM, segT, Nh, struct('zerolat', zl));
    Nh0 = O.Nh0; O.split = struct();
    bar_b = run(base, Nh0, 1); inf_b = run(base, P.Nh_inf, 1);
    for j = 1:4
      m = O.best{j}; if strcmp(m, 'b'), continue; end
      impl = base; impl{j} = m; Nh = nh_for(impl, pf, P, C, smean);
      bar_m = run(impl, Nh, 1); inf_m = run(impl, P.Nh_inf, 1);
      dHp = O.single.(m).te.H - O.base_te.H; dHbar = bar_m.H - bar_b.H; dHinf = inf_m.H - inf_b.H;
      O.split.(m) = [dHinf, dHbar, dHp, dHinf - dHbar, dHbar - dHp];
      printf('C%d seed%d %s Nh=%d Nh0=%d inf=%.4f bar=%.4f p=%.4f\n', ip, seed, m, Nh, Nh0, dHinf, dHbar, dHp); fflush(stdout);
    end
    save('-v7', fn, 'O');
  end
end
end
