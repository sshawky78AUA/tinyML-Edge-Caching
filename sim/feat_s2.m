function X = feat_s2(tr, P, items, lastIdx, kmin, t, cntH)
% FEAT_S2  Features of stage S2 for the given items at time t: log gaps
% between the last P.nd requests of each item inside the history, log size,
% and log request count in the history. lastIdx(f) is the index of the
% latest request of item f that has arrived; requests older than kmin are
% outside the history and are treated as missing.
items = items(:); n = numel(items); big = log1p(1e5);
ts = 1; if isfield(P, 'tscale'), ts = P.tscale; end
G = big*ones(n, P.nd);
k = lastIdx(items); tprev = t*ones(n,1); v = k >= kmin & k > 0;
for d = 1:P.nd
  if ~any(v), break; end
  kv = k(v); tk = tr.t(kv);
  G(v,d) = log1p((tprev(v) - tk)*ts); tprev(v) = tk; k(v) = tr.prevIdx(kv);
  v = v & k >= kmin & k > 0;
end
X = [G log(tr.s(items)) log1p(cntH(items))];
end
