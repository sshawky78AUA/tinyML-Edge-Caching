function tr = gen_trace(P, seed)
% GEN_TRACE  Synthetic request trace with user mobility, Zipf popularity,
% popularity drift, and per-user item successions.
%
% Users move on a ring of P.Nc cells; cell 1 is the coverage area of the EN.
% A user issues requests (Poisson, rate P.ru) only while inside cell 1.
% Each request is, with probability P.beta, a successor of the user's
% previous item (fixed successor table), and otherwise a draw from a Zipf
% law over a ranking that is partially reshuffled every P.Tdrift seconds.
%
% Output struct tr: t (K x 1), u, f, s (item sizes, F x 1), and the
% mobility log mob = [t, u, cell] sorted by time.

rand('state', seed); randn('state', seed);
F = P.F; U = P.U; T = P.T;

% item sizes (bytes), lognormal, clipped
s = round(exp(log(P.size_median) + P.size_sigma*randn(F,1)));
s = min(max(s, P.size_min), P.size_max);

% successor table: each item has 3 successors with probs P.succ_p
w0 = (1:F)'.^(-P.alpha); cdf0 = cumsum(w0)/sum(w0); perm0 = randperm(F)';
succ = zeros(F,3);
for a = 1:F*3, succ(a) = perm0(find(rand <= cdf0, 1)); end  % popularity-biased successors

% ---- mobility ----
mobT = []; mobU = []; mobC = [];
reqT = []; reqU = [];
for u = 1:U
  c = randi(P.Nc); t = 0; dir = 2*(rand < 0.5) - 1;
  mobT(end+1,1) = 0; mobU(end+1,1) = u; mobC(end+1,1) = c; %#ok<AGROW>
  while t < T
    dwell = sum(-log(rand(P.dwell_shape,1)))*P.dwell_mean/P.dwell_shape;  % gamma dwell
    t2 = min(t + dwell, T);
    if c == 1
      n = poissrnd_local(P.ru*(t2 - t));
      if n > 0
        reqT = [reqT; t + (t2 - t)*rand(n,1)]; %#ok<AGROW>
        reqU = [reqU; u*ones(n,1)];           %#ok<AGROW>
      end
    end
    t = t2;
    if t >= T, break; end
    if rand > P.persist, dir = -dir; end   % keep direction with prob. P.persist
    c = c + dir;
    c = mod(c - 1, P.Nc) + 1;
    mobT(end+1,1) = t; mobU(end+1,1) = u; mobC(end+1,1) = c; %#ok<AGROW>
  end
end
[mobT, o] = sort(mobT); mob = [mobT, mobU(o), mobC(o)];
[reqT, o] = sort(reqT); reqU = reqU(o);
K = numel(reqT);

% ---- popularity with drift ----
w = (1:F)'.^(-P.alpha); cdf = cumsum(w)/sum(w);
rank2item = randperm(F)';
nextDrift = P.Tdrift;
lastItem = zeros(U,1);
f = zeros(K,1);
for k = 1:K
  while reqT(k) >= nextDrift
    % reshuffle: each of the top P.drift_top ranks swaps with a random rank
    for r = 1:P.drift_top
      if rand < P.drift_p
        r2 = randi(F); tmp = rank2item(r); rank2item(r) = rank2item(r2); rank2item(r2) = tmp;
      end
    end
    nextDrift = nextDrift + P.Tdrift;
  end
  u = reqU(k);
  if lastItem(u) > 0 && rand < P.beta
    x = rand; j = find(x <= cumsum(P.succ_p), 1);
    f(k) = succ(lastItem(u), j);
  else
    f(k) = rank2item(find(rand <= cdf, 1));
  end
  lastItem(u) = f(k);
end

tr.t = reqT; tr.u = reqU; tr.f = f; tr.s = s; tr.mob = mob;
tr.K = K; tr.F = F; tr.U = U; tr.succ = succ;
end

function n = poissrnd_local(m)
% Poisson sample by inversion (small means) or normal approximation.
if m < 50
  L = exp(-m); n = 0; p = rand;
  while p > L, n = n + 1; p = p*rand; end
else
  n = max(0, round(m + sqrt(m)*randn));
end
end
