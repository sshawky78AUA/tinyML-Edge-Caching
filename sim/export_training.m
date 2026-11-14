function D = export_training(tr, P, t0, t1, outprefix)
% EXPORT_TRAINING  Replay requests in [t0, t1) through a request history of
% length P.Nh_inf and write the training examples of every stage to CSV
% files <outprefix>_s{1..4}.csv (features followed by the label).
rand('state', 7);
Nh = P.Nh_inf; F = tr.F;
ks = find(tr.t >= t0 & tr.t < t1);
D1 = []; D2 = []; D3 = []; D4 = [];
lastIdx = zeros(F,1); cntH = zeros(F,1);
% preload history with all requests before t0
kpre = find(tr.t < t0);
for k = kpre(:)'
  lastIdx(tr.f(k)) = k; cntH(tr.f(k)) = cntH(tr.f(k)) + 1;
  if k - Nh >= 1, cntH(tr.f(k-Nh)) = cntH(tr.f(k-Nh)) - 1; end
end
ms = mob_init(tr, P, t0);
nextEp1 = ceil(t0/P.T1)*P.T1; nextEp4 = ceil(t0/P.T4)*P.T4;
for k = ks(:)'
  t = tr.t(k);
  % periodic epochs before this request
  while nextEp1 <= t
    kcur = k - 1; kmin = max(1, kcur - Nh + 1);
    [items, X] = feat_s1(tr, P, kmin, kcur, nextEp1);
    if ~isempty(items)
      nx = tr.nextIdx(lastIdx(items)); tn = inf(size(nx)); okn = nx <= tr.K; tn(okn) = tr.t(nx(okn));
      D1 = [D1; X, (tn <= nextEp1 + P.W1)]; %#ok<AGROW>
    end
    nextEp1 = nextEp1 + P.T1;
  end
  while nextEp4 <= t
    ms = mob_advance(ms, tr, P, nextEp4);
    [users, X] = feat_s4(P, ms, nextEp4);
    if ~isempty(users)
      y = zeros(numel(users),1);
      for a = 1:numel(users)
        e = find(tr.mobU == users(a) & tr.mobT > nextEp4 & tr.mobT <= nextEp4 + P.W4);
        y(a) = any(tr.mobC(e) == 1);
      end
      D4 = [D4; X, y]; %#ok<AGROW>
    end
    nextEp4 = nextEp4 + P.T4;
  end
  % request k enters the history
  f = tr.f(k); lastIdx(f) = k; cntH(f) = cntH(f) + 1;
  if k - Nh >= 1, cntH(tr.f(k-Nh)) = cntH(tr.f(k-Nh)) - 1; end
  kmin = max(1, k - Nh + 1);
  % S2: the requested item, plus random items of the history every 5th request
  its = f;
  if mod(k,5) == 0, its = [its; tr.f(kmin - 1 + randi(k - kmin + 1, 3, 1))]; end
  X = feat_s2(tr, P, its, lastIdx, kmin, t, cntH);
  nx = tr.nextIdx(lastIdx(its)); tn = inf(size(nx)); okn = nx <= tr.K; tn(okn) = tr.t(nx(okn));
  D2 = [D2; X, (tn <= t + P.W2)]; %#ok<AGROW>
  % S3 examples are written by export_s3c.m
end
D = {D1, D2, D3, D4};
for j = [1 2 4]
  csvwrite(sprintf('%s_s%d.csv', outprefix, j), D{j});
end
end
