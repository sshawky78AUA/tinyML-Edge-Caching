function V = validate_sim(P)
% VALIDATE_SIM  Two checks of the simulator (Section V-E):
%  (1) LRU under independent Zipf requests with equal sizes against Che's
%      approximation;  (2) the processor queue under Poisson triggers with a
%      constant service time against the Pollaczek-Khinchine mean wait.
rand('state', 3);
% ---- (1) Che's approximation ----
F = 2000; alpha = 0.8; K = 200000; w = (1:F)'.^(-alpha); q = w/sum(w); cdf = cumsum(q);
f = arrayfun(@(x) find(x <= cdf, 1), rand(K,1));
V.che = [];
for Cn = [20 50 100 200 400]
  ts = zeros(F,1); inC = false(F,1); n = 0; hit = 0;
  for k = 1:K
    x = f(k);
    if inC(x), hit = hit + (k > K/4);
    else
      if n == Cn, c = find(inC); [~, m] = min(ts(c)); inC(c(m)) = false; n = n - 1; end
      inC(x) = true; n = n + 1;
    end
    ts(x) = k;
  end
  Hsim = hit/(0.75*K);
  tc = fzero(@(T) sum(1 - exp(-q*T)) - Cn, [Cn, 1e9]);     % characteristic time
  Hche = sum(q.*(1 - exp(-q*tc)));
  V.che(end+1,:) = [Cn, Hsim, Hche, abs(Hsim - Hche)];
end
% ---- (2) Pollaczek-Khinchine ----
V.pk = [];
tau = 1e-3;
for rho = [0.2 0.5 0.8]
  lam = rho/tau; n = 200000; a = cumsum(-log(rand(n,1))/lam);
  cfree = 0; wait = zeros(n,1);
  for i = 1:n, s = max(a(i), cfree); wait(i) = s - a(i); cfree = s + tau; end
  Wsim = mean(wait(n/10:end)); Wpk = rho*tau/(2*(1 - rho));
  V.pk(end+1,:) = [rho, Wsim, Wpk, abs(Wsim - Wpk)/Wpk];
end
end
