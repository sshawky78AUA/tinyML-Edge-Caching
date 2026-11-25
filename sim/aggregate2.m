function aggregate2(tag, seeds)
% AGGREGATE2  Collect res2<tag>_C*_seed*.mat into results2<tag>.json.
P = params_default(); if strcmp(tag, 'mt'), P.modeldir = '../models/mt/'; end
names = {'s1_lin','s1_mlp_s','s1_mlp_l','s2_lin','s2_mlp_s','s2_mlp_l', ...
  's3_lin','s3_mlp_s','s3_mlp_l','s4_lin','s4_mlp_s','s4_mlp_l'};
A = struct(); [~, pfs, mdl] = load_all(P, 1);
for ip = 1:4
  c = struct(); c.seeds = {}; pf = pfs{ip}; c.name = pf.name;
  for a = 1:numel(names)                       % latency and energy of one inference (Fig. 2)
    n = names{a}; cy = pf.cost.(n);
    c.inf.(n) = struct('cpu', cy(1), 'npu', cy(2), 'tau', cy(1)/pf.phi + cy(2)/pf.phin, ...
      'E', pf.ecpu*cy(1) + pf.enpu*cy(2), 'macs', mdl.(n).macs, 'bytes', mdl.(n).bytes);
  end
  for s = seeds
    fn = sprintf('results/res2%s_C%d_seed%d.mat', tag, ip, s);
    if ~exist(fn, 'file'), continue; end
    Q = load(fn); O = Q.O; b = O.base_te; r = struct();
    r.seed = s; r.Nh0 = O.Nh0; r.base = pick(b); r.best = O.best; r.comb = O.comb; r.combH = O.combH;
    r.m2H = O.m2H; r.bestA = O.bestA; r.split = O.split; r.I = O.I;
    for a = 1:numel(names)
      n = names{a}; if ~isfield(O.single, n), continue; end
      x = O.single.(n); e = struct('feas', x.feas, 'Nh', x.Nh);
      if x.feas && isfield(x, 'va'), e.dHva = x.va.H - O.base_va.H; e.Gva = O.base_va.Esys - x.va.Esys; e.rhova = x.va.rho; e.dEENva = x.va.EEN - O.base_va.EEN; e.Kva = O.base_va.K; end
      if x.feas && isfield(x, 'te'), e = diffs(e, x.te, b); end
      r.single.(n) = e;
    end
    r.comb_res = diffs(struct(), O.comb_te, b); r.combH_res = diffs(struct(), O.combH_te, b);
    r.A_res = diffs(struct(), O.A_te, b); r.behind = struct();
    if isfield(O, 'behind')
      fnb = fieldnames(O.behind);
      for a = 1:numel(fnb)
        x = O.behind.(fnb{a}); e = struct('feas', x.feas);
        if isfield(x, 'va'), e.dHva = x.va.H - O.A_va.H; end
        if isfield(x, 'te'), e.dHte = x.te.H - O.A_te.H; e.dEsys = x.te.Esys - O.A_te.Esys; e.dD = x.te.D - O.A_te.D; end
        r.behind.(fnb{a}) = e;
      end
    end
    c.seeds{end+1} = r;
  end
  A.(sprintf('C%d', ip)) = c;
end
fid = fopen(sprintf('results/results2%s.json', tag), 'w'); fprintf(fid, '%s', jsonencode(A)); fclose(fid);
end
function e = diffs(e, t, b)
e.H = t.H; e.HB = t.HB; e.dH = t.H - b.H; e.dHB = t.HB - b.HB; e.G = b.Esys - t.Esys;
e.dEsys = t.Esys - b.Esys; e.dEEN = t.EEN - b.EEN; e.dElink = t.Elink - b.Elink;
e.dD = t.D - b.D; e.dBrel = (t.B - b.B)/b.B; e.rho = t.rho; e.drop = t.drop; e.Npf = t.Npf;
e.EEN = t.EEN; e.Esys = t.Esys; e.Einf = t.Einf; e.D = t.D; e.Hl = t.Hl(:)';
end
function p = pick(b)
p = struct('H', b.H, 'HB', b.HB, 'B', b.B, 'D', b.D, 'EEN', b.EEN, 'Einf', b.Einf, 'Eslp', b.Eslp, ...
  'Elink', b.Elink, 'Esys', b.Esys, 'rho', b.rho, 'K', b.K, 'lambda', b.lambda, 'Hl', b.Hl(:)');
end
