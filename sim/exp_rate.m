function exp_rate(ips, gammas, seed)
% EXP_RATE  Break-even study (RQ4): time-scale the trace by gamma, run the
% baseline and each selected single-stage configuration on every platform,
% and store node energy and backhaul bytes so that G_j can be evaluated for
% any backhaul energy e_b afterwards (E_link is linear in e_b).
P = params_default(); load('trace_s1p.mat'); tr0 = tr;
S = load('results_b2.mat'); b2 = S.b2;
kt = find(tr0.t >= tr0.Tva); C = P.Cfrac*sum(tr0.s(unique(tr0.f(kt))));
smean = mean(tr0.s(tr0.f));
for ip = ips
  [WM, pfs] = load_all(P, seed); pf = pfs{ip};
  Q = load(sprintf('results_C%d_seed%d.mat', ip, seed)); best = Q.O.best;
  out = struct('gamma', gammas, 'runs', {{}});
  for g = gammas
    tr = tr0; tr.t = tr0.t(1) + (tr0.t - tr0.t(1))/g; tr.mobT = tr0.t(1) + (tr0.mobT - tr0.t(1))/g;
    Tva = tr0.t(1) + (tr0.Tva - tr0.t(1))/g; Tend = tr0.t(1) + (tr0.Tend - tr0.t(1))/g;
    seg = [Tva - 4*3600/g, Tva, Tend];
    base = {'b','b','b','b'}; rr = struct();
    Nh = nh_for(base, pf, P, C, smean);
    rr.base = run_sim(tr, P, struct('impl', {base}, 's2base', b2, 'C', C), pf, WM, seg, Nh, struct());
    for j = 1:4
      for a = {sprintf('s%d_mlp_s', j), best{j}}
        m = a{1}; if strcmp(m, 'b') || isfield(rr, m), continue; end
        impl = base; impl{j} = m; [Nh, feas] = nh_for(impl, pf, P, C, smean);
        if ~feas, continue; end
        rr.(m) = run_sim(tr, P, struct('impl', {impl}, 's2base', b2, 'C', C), pf, WM, seg, Nh, struct());
        printf('C%d g=%g %s dH=%.4f drop=%s rho=%.3g\n', ip, g, m, rr.(m).H - rr.base.H, mat2str(rr.(m).drop), rr.(m).rho); fflush(stdout);
      end
    end
    out.runs{end+1} = rr;
  end
  save('-v7', sprintf('results_rate_C%d.mat', ip), 'out');
end
end
