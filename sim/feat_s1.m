function [items, X, ewma] = feat_s1(tr, P, kmin, kcur, t)
% FEAT_S1  Features of stage S1 at time t from the history H = requests
% kmin..kcur. Returns the tracked items (distinct items in H), their
% feature matrix X (log window counts and log history count), and the
% EWMA baseline score.
idx = kmin:kcur;
if isempty(idx), items = zeros(0,1); X = zeros(0, P.nw+1); ewma = zeros(0,1); return; end
f = tr.f(idx); tt = tr.t(idx);
[items, ~, j] = unique(f);
bin = floor((t - tt)/P.T1) + 1;              % 1 = most recent interval
ok = bin <= P.nw;
C = accumarray([j(ok) bin(ok)], 1, [numel(items) P.nw]);
tot = accumarray(j, 1, [numel(items) 1]);
X = [log1p(C) log1p(tot)];
ewma = C * (P.rho_ewma.^(0:P.nw-1))';
end
