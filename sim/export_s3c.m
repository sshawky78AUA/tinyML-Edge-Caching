function export_s3c(outprefix, t0, t1, tracefile)
% EXPORT_S3C  Training examples of S3 in [t0, t1): for every request k whose
% user makes a later request, one row per candidate of feat_s3 (features,
% then label = 1 if the candidate is the user's next item). The history has
% length P.Nh_inf and is preloaded with the requests before t0.
P = params_default(); if nargin < 4, tracefile = 'trace_s1p.mat'; end; load(tracefile); F = tr.F; Nh = P.Nh_inf;
lastIdx = zeros(F,1); cntH = zeros(F,1); T = zeros(F, F, 'int16');
kpre = find(tr.t < t0);
if ~isempty(kpre)
  k0 = kpre(max(1, end-Nh+1));
  for k = kpre(max(1, end-Nh+1):end)'
    lastIdx(tr.f(k)) = k; cntH(tr.f(k)) = cntH(tr.f(k)) + 1;
    j = tr.prevU(k); if j >= k0, T(tr.f(j), tr.f(k)) = T(tr.f(j), tr.f(k)) + 1; end
  end
end
ks = find(tr.t >= t0 & tr.t < t1); D = zeros(numel(ks)*P.Nc3, 8); n = 0;
for k = ks(:)'
  f = tr.f(k); lastIdx(f) = k; cntH(f) = cntH(f) + 1;
  j = tr.prevU(k); if j >= k - Nh + 1 && j > 0, T(tr.f(j), f) = T(tr.f(j), f) + 1; end
  if k - Nh >= 1
    e = k - Nh; cntH(tr.f(e)) = cntH(tr.f(e)) - 1;
    n2 = tr.nextU(e); if n2 > 0 && n2 < k, T(tr.f(e), tr.f(n2)) = T(tr.f(e), tr.f(n2)) - 1; end
  end
  if tr.nextU(k) > 0
    kmin = max(1, k - Nh + 1);
    [c, X] = feat_s3(tr, P, f, T, cntH, lastIdx, kmin, tr.t(k));
    m = numel(c); if m == 0, continue; end
    D(n+1:n+m, :) = [X, double(c == tr.f(tr.nextU(k))), k*ones(m,1)]; n = n + m;
  end
end
csvwrite(sprintf('%s_s3.csv', outprefix), D(1:n, 1:7));
csvwrite(sprintf('%s_s3_req.csv', outprefix), D(1:n, 8));
printf('%s: %d rows, %d requests, positives %.3f\n', outprefix, n, numel(unique(D(1:n,8))), mean(D(1:n,7)));
end
