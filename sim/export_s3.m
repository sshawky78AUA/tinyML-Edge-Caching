function export_s3(outprefix, t0, t1)
% EXPORT_S3  Rebuild the S3 vocabulary (P.Fe most frequent training items)
% in trace_s1p.mat and write the S3 examples of [t0, t1) to <outprefix>_s3.csv.
P = params_default(); load('trace_s1p.mat');
ktr = find(tr.t < tr.Ttr); cnt = accumarray(tr.f(ktr), 1, [tr.F 1]);
[~, o] = sort(cnt, 'descend'); Fe = min(P.Fe, tr.F);
vocab = zeros(tr.F,1); vocab(o(1:Fe)) = 1:Fe; tr.vocab = vocab; tr.vocabItems = o(1:Fe);
save('-v7', 'trace_s1p.mat', 'tr');
if nargin < 2, return; end
ks = find(tr.t >= t0 & tr.t < t1); D3 = zeros(numel(ks), P.Q3 + 1); n = 0;
for k = ks(:)'
  kmin = max(1, k - P.Nh_inf + 1);
  if tr.nextU(k) > 0
    v = tr.vocab(tr.f(tr.nextU(k))); if v == 0, v = P.Fe + 1; end
    n = n + 1; D3(n,:) = [feat_s3(tr, P, k, kmin), v];
  end
end
csvwrite(sprintf('%s_s3.csv', outprefix), D3(1:n,:));
end
