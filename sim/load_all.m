function [WM, pf, mdl] = load_all(P, seed)
% LOAD_ALL  Load model weights (one training seed), cycle counts, Vela
% estimates, and build the platform profiles.
names = {'s1_lin','s1_mlp_s','s1_mlp_l','s2_lin','s2_mlp_s','s2_mlp_l', ...
         's3_lin','s3_mlp_s','s3_mlp_l','s4_lin','s4_mlp_s','s4_mlp_l'};
md = '../models/'; if isfield(P, 'modeldir'), md = P.modeldir; end
sm = jsondecode(fileread(sprintf('%ssummary_seed%d.json', md, seed)));
vl = jsondecode(fileread('../models/vela_summary.json'));
cyc = jsondecode(fileread('../fw/cycles.json'));
fn = fieldnames(cyc); for a = 1:numel(fn), v = cyc.(fn{a}); cyc.(fn{a}) = v(1); end
for a = 1:numel(names)
  n = names{a};
  wf = sprintf('%sweights/%s_seed%d.mat', md, n, seed);
  if ~exist(wf, 'file'), wf = sprintf('../models/weights/%s_seed%d.mat', n, seed); end   % stage not trained on this trace
  W = load(wf);
  W.kind = strtrim(W.kind); WM.(n) = W;
  r = sm(strcmp({sm.key}, n));
  if isempty(r), sm0 = jsondecode(fileread(sprintf('../models/summary_seed%d.json', seed))); r = sm0(strcmp({sm0.key}, n)); end
  mdl.(n).macs = r.macs; mdl.(n).bytes = r.tflite_bytes;
  wid = 0; for l = 1:W.nlayers, wid = max(wid, size(W.(sprintf('W%d',l)),2)); end
  if isfield(W,'E'), wid = max(wid, size(W.E,2)*3); end
  mdl.(n).ram = 2*wid + 256;            % activation buffers (int8) + scratch
  n2 = n; if strncmp(n, 's3', 2), n2 = ['s2' n(3:end)]; end   % S3 scorer = S2 architecture
  v = vl(strcmp({vl.cfg}, 'ethos-u55-128') & strcmp({vl.net}, n2));
  vela.(n) = v.cycles_total;
end
pf = platforms(P, cyc, vela, mdl);
end
