function [cand, X, cnt] = feat_s3(tr, P, f, T, cntH, lastIdx, kmin, t)
% FEAT_S3  Candidates and features of stage S3 for a user whose latest item
% is f. The candidates are the P.Nc3 items that most often followed f in
% the request history (T(f,c) = number of times a user requested c right
% after f, both requests inside the history). Features of candidate c: log
% transition count, log number of transitions out of f, transition share,
% log request count of c in the history, log size of c, and log time since
% the last request of c (a large value if c has no request in the history).
row = double(T(f,:)); nz = find(row > 0);
if isempty(nz), cand = []; X = zeros(0, 6); cnt = []; return; end
[cs, o] = sort(row(nz), 'descend'); n = min(P.Nc3, numel(nz));
cand = nz(o(1:n)); cand = cand(:); cnt = cs(1:n); cnt = cnt(:); nf = sum(row);
li = lastIdx(cand); rec = log1p(1e5)*ones(n,1); ok = li >= kmin & li > 0;
ts = 1; if isfield(P, "tscale"), ts = P.tscale; end
rec(ok) = log1p((t - tr.t(li(ok)))*ts);
X = [log1p(cnt), log1p(nf)*ones(n,1), cnt/nf, log1p(cntH(cand)), log(tr.s(cand)), rec];
end
