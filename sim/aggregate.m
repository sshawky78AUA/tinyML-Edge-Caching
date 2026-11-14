function aggregate(seeds)
% AGGREGATE  Collect the benchmark results into results_summary.json.
P = params_default(); names = {'s1_lin','s1_mlp_s','s1_mlp_l','s2_lin','s2_mlp_s','s2_mlp_l', ...
  's3_lin','s3_mlp_s','s3_mlp_l','s4_lin','s4_mlp_s','s4_mlp_l'};
[~, pfs, mdl] = load_all(P, 1);
A = struct();
for ip = 1:4
  pf = pfs{ip}; c = struct(); c.name = pf.name;
  c.prof = struct('phi', pf.phi, 'ecpu', pf.ecpu, 'enpu', pf.enpu, 'Pslp', pf.Pslp, ...
    'twk', pf.twk, 'Ewk', pf.Ewk, 'Mram', pf.Mram, 'Mfl', pf.Mfl);
  for a = 1:numel(names)
    n = names{a}; cy = pf.cost.(n);
    c.inf.(n) = struct('cpu', cy(1), 'npu', cy(2), 'tau', cy(1)/pf.phi + cy(2)/pf.phin, ...
      'E', pf.ecpu*cy(1) + pf.enpu*cy(2), 'macs', mdl.(n).macs, 'bytes', mdl.(n).bytes);
  end
  for s = seeds
    Q = load(sprintf('results_C%d_seed%d.mat', ip, s)); O = Q.O;
    r = struct(); r.Nh0 = O.Nh0; b = O.base_te;
    r.base = pick(b);
    for a = 1:numel(names)
      n = names{a}; x = O.single.(n); e = struct('feas', x.feas, 'Nh', x.Nh);
      if x.feas
        t = x.te; e.dH = t.H - b.H; e.dHB = t.HB - b.HB; e.G = b.Esys - t.Esys;
        e.dEEN = t.EEN - b.EEN; e.dElink = t.Elink - b.Elink; e.dD = b.D - t.D; e.dB = (b.B - t.B)/b.B;
        e.rho = t.rho; e.drop = t.drop; e.nd = t.nd; e.Npf = t.Npf; e.Hl = t.Hl(:)';
        e.dHva = x.va.H - O.base_va.H; e.Gva = O.base_va.Esys - x.va.Esys;
      end
      r.single.(n) = e;
    end
    r.best = O.best; r.comb = O.comb; r.comb_Nh = O.comb_Nh;
    r.comb_res = pick(O.comb_te); r.comb_dH = O.comb_te.H - b.H; r.comb_G = b.Esys - O.comb_te.Esys;
    r.I = O.I; r.split = O.split;
    c.seeds{s} = r;
  end
  A.(sprintf('C%d', ip)) = c;
end
S = load('results_validation.mat'); A.validation = S.V;
S = load('results_belady.mat'); A.belady = S.HB;
S = load('results_b2.mat'); A.b2 = S.b2;
fid = fopen('results_summary.json', 'w'); fprintf(fid, '%s', jsonencode(A)); fclose(fid);
end

function p = pick(R)
p = struct('H', R.H, 'HB', R.HB, 'B', R.B, 'D', R.D, 'EEN', R.EEN, 'Einf', R.Einf, 'Eslp', R.Eslp, ...
  'Elink', R.Elink, 'Esys', R.Esys, 'rho', R.rho, 'nd', R.nd, 'drop', R.drop, 'Hl', R.Hl(:)', 'K', R.K, 'lambda', R.lambda);
end
