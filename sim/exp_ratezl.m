function exp_ratezl(ip, g, seed)
% EXP_RATEZL  At rate scaling g, the hit-ratio-oriented pipeline and the
% baseline with zero service times, to separate the latency loss from the
% effect of the rate itself.
P = params_default(); load('trace_s1p.mat'); tr0 = tr; S = load('results_b2.mat'); b2 = S.b2;
kt = find(tr0.t >= tr0.Tva); C = P.Cfrac*sum(tr0.s(unique(tr0.f(kt)))); smean = mean(tr0.s(tr0.f));
[WM, pfs] = load_all(P, seed); pf = pfs{ip}; Q = load(sprintf('res2_C%d_seed%d.mat', ip, seed)); O = Q.O;
tr = tr0; tr.t = tr0.t(1) + (tr0.t - tr0.t(1))/g; tr.mobT = tr0.t(1) + (tr0.mobT - tr0.t(1))/g;
Tva = tr0.t(1) + (tr0.Tva - tr0.t(1))/g; Tend = tr0.t(1) + (tr0.Tend - tr0.t(1))/g; seg = [Tva - 4*3600/g, Tva, Tend];
P2 = P; P2.T1 = P.T1/g; P2.T4 = P.T4/g; P2.tscale = g; P2.tref = tr0.t(1);
base = {'b','b','b','b'};
Rb = run_sim(tr, P2, struct('impl', {base}, 's2base', b2, 'C', C), pf, WM, seg, nh_for(base, pf, P, C, smean), struct('zerolat', 1));
Rh = run_sim(tr, P2, struct('impl', {O.combH}, 's2base', b2, 'C', C), pf, WM, seg, nh_for(O.combH, pf, P, C, smean), struct('zerolat', 1));
printf('C%d g=%g zerolat combH dH=%.4f\n', ip, g, Rh.H - Rb.H);
save('-v7', sprintf('ratezl_C%d_g%g.mat', ip, g), 'Rb', 'Rh');
end
