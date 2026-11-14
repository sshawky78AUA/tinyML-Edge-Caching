function tr = prep_trace(tr, P)
% PREP_TRACE  Precompute trace-only index structures used by the simulator
% and by the feature extractors. All structures depend only on the trace,
% never on cache decisions (Assumption 3 of the paper).
K = tr.K; F = tr.F;
prevIdx = zeros(K,1); nextIdx = (K+1)*ones(K,1); last = zeros(F,1);
prevU = zeros(K,1); lastU = zeros(tr.U,1); nextU = zeros(K,1);
for k = 1:K
  f = tr.f(k); u = tr.u(k);
  prevIdx(k) = last(f); if last(f) > 0, nextIdx(last(f)) = k; end
  last(f) = k;
  prevU(k) = lastU(u); if lastU(u) > 0, nextU(lastU(u)) = k; end
  lastU(u) = k;
end
tr.prevIdx = prevIdx; tr.nextIdx = nextIdx; tr.prevU = prevU; tr.nextU = nextU;

% split instants
tr.Ttr = P.frac_tr*P.T; tr.Tva = (P.frac_tr + P.frac_va)*P.T; tr.Tend = P.T;

% vocabulary for S3: the Fe most frequent items of the training part
ktr = find(tr.t < tr.Ttr);
cnt = accumarray(tr.f(ktr), 1, [F 1]);
[~, o] = sort(cnt, 'descend');
vocab = zeros(F,1); vocab(o(1:P.Fe)) = 1:P.Fe;   % 0 = out of vocabulary
tr.vocab = vocab; tr.vocabItems = o(1:P.Fe);

% delays and link energies per item
tr.da_f = P.da + tr.s/P.Ra; tr.db_f = P.db + tr.s/P.Rb;

% mobility: per-user cell at any time via the sorted log
tr.mobT = tr.mob(:,1); tr.mobU = tr.mob(:,2); tr.mobC = tr.mob(:,3);
end
