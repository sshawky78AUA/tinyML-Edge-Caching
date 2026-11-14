function R = run_sim(tr, P, cfg, pf, WM, seg, Nh, opt)
% RUN_SIM  Event-driven simulation of the edge cache node (Section III).
%  cfg.impl : 1x4 cell, 'b' (baseline) or a model key, e.g. 's2_mlp_s'
%  cfg.s2base: 'lru' | 'lfu' | 'arc'   (baseline b_2)
%  pf       : platform profile (platforms.m); WM: struct of model weights
%  seg      : [t0 te t1]  simulate from t0, measure on [te, t1)
%  Nh       : history length;   opt.zerolat: decisions take no time
%  C (bytes) is passed in cfg.C.
% Returns metrics on the measurement window (paper eqs. (19)-(26)).
if nargin < 8, opt = struct(); end
zl = isfield(opt, 'zerolat') && opt.zerolat;
rand('state', 11);
F = tr.F; C = cfg.C; imp = cfg.impl; base2 = cfg.s2base;
t0 = seg(1); te = seg(2); t1 = seg(3);
learned = ~strcmp(imp, 'b');

% ---------- state ----------
L = zeros(F,1);          % 0 absent; 1 cached (T1 for ARC); 2 cached T2; 3 ghost B1; 4 ghost B2
ts = zeros(F,1); fr = zeros(F,1); used = 0; bL = zeros(4,1); parc = 0;
inflight = false(F,1);
lastIdx = zeros(F,1); cntH = zeros(F,1);
useT = any(~strcmp(imp(3:4), 'b'));          % transition counts needed by S3/S4
if useT, T = zeros(F, F, 'int16'); else, T = zeros(1, 1, 'int16'); end
kpre = find(tr.t < t0); k0 = kpre(max(1, end-Nh+1));
for k = kpre(max(1, end-Nh+1):end)'
  lastIdx(tr.f(k)) = k; cntH(tr.f(k)) = cntH(tr.f(k)) + 1;
  j = tr.prevU(k); if useT && j >= k0, T(tr.f(j), tr.f(k)) = T(tr.f(j), tr.f(k)) + 1; end
end
ms = mob_init(tr, P, t0);
cfree = 0;                                    % time the processor becomes free
evT = []; evTy = []; evA = {};                % pending events
nxt1 = ceil(t0/P.T1)*P.T1; nxt4 = ceil(t0/P.T4)*P.T4;

% ---------- accumulators ----------
nd = zeros(1,4); drop = zeros(1,4); Einf = 0; busy = 0; Spf = 0; Npf = 0;
hit = []; sz = []; tk = [];

ks = find(tr.t >= t0 & tr.t < t1)';
for k = [ks, 0]
  if k > 0, tnow = tr.t(k); else, tnow = t1; end
  % ---- process epochs and pending events up to tnow ----
  while true
    [tev, iev] = min([evT, inf]);
    tn = min([tev, nxt1, nxt4]);
    if tn > tnow || tn >= t1, break; end
    kcur = (k > 0)*(k - 1) + (k == 0)*ks(end);
    kmin = max(1, kcur - Nh + 1);
    if tn == tev
      ty = evTy(iev); a = evA{iev}; evT(iev) = []; evTy(iev) = []; evA(iev) = [];
      if ty == 1, candidates(a, tev, kmin); else, insert_item(a(1), a(2), a(3), tev, kmin); end
    elseif tn == nxt1
      stage1(nxt1, kmin, kcur); nxt1 = nxt1 + P.T1;
    else
      ms = mob_advance(ms, tr, P, nxt4);
      if learned(4), stage4(nxt4, kmin, kcur); end
      nxt4 = nxt4 + P.T4;
    end
  end
  if k == 0, break; end

  % ---- request k ----
  f = tr.f(k); t = tnow; meas = t >= te;
  h = (L(f) == 1 || L(f) == 2);
  if meas, hit(end+1) = h; sz(end+1) = tr.s(f); tk(end+1) = t; end %#ok<AGROW>
  lastIdx(f) = k; cntH(f) = cntH(f) + 1;
  if useT
    j = tr.prevU(k); if j >= k - Nh + 1, T(tr.f(j), f) = T(tr.f(j), f) + 1; end
  end
  if k - Nh >= 1
    e = k - Nh; cntH(tr.f(e)) = cntH(tr.f(e)) - 1;
    if useT, n2 = tr.nextU(e); if n2 > 0 && n2 < k, T(tr.f(e), tr.f(n2)) = T(tr.f(e), tr.f(n2)) - 1; end, end
  end
  kmin = max(1, k - Nh + 1);
  if h
    ts(f) = t; fr(f) = fr(f) + 1;
    if strcmp(base2, 'arc') && L(f) == 1, setL(f, 2); end
  elseif ~inflight(f)
    tgt = arc_ghost(f);                         % ARC adaptation at request time
    [ok, c] = release(2, t, s2cycles(), tr.db_f(f));
    if learned(2) && ok
      p = model_eval(WM.(imp{2}), feat_s2(tr, P, f, lastIdx, kmin, t, cntH));
      if p >= P.theta2, push(2, t + tr.db_f(f), [f 1 tgt]); inflight(f) = true; end
    else
      push(2, t + tr.db_f(f), [f 0 tgt]); inflight(f) = true;
    end
  end
  if learned(3)
    [cand, sel, cyc] = s3_select(f, kmin, t);
    [ok, c] = release(3, t, cyc, inf);
    if ok && ~isempty(sel), push(1, c, sel(:)'); end
  end
end

% ---------- metrics ----------
hit = hit(:); sz = sz(:); tk = tk(:); Kte = numel(hit); Tte = t1 - te;
fk = tr.f(ks(tr.t(ks) >= te));
R.H = mean(hit); R.HB = sum(hit.*sz)/sum(sz);
R.B = sum((1-hit).*sz) + Spf; R.Spf = Spf; R.Npf = Npf;
dk = tr.da_f(fk) + (1-hit).*tr.db_f(fk);
Ek = P.ea*sz + (1-hit)*P.eb.*sz;
R.D = mean(dk); R.Elink = sum(Ek) + P.eb*Spf;
R.Einf = Einf; R.Eslp = Tte*pf.Pslp; R.EEN = Einf + R.Eslp;
R.Esys = R.EEN + R.Elink;
R.nd = nd; R.drop = drop; R.K = Kte; R.lambda = Kte/Tte;
nw = ceil(Tte/P.Tw); w = min(nw, floor((tk - te)/P.Tw) + 1);
R.Hl = accumarray(w, hit, [nw 1]) ./ max(1, accumarray(w, 1, [nw 1]));
R.rho = busy/Tte; R.bytesB = sum((1-hit).*sz) + Spf; R.bytesA = sum(sz);

% =================== nested helpers ===================
  function [ok, c] = release(j, q, cyc, dl)
    % FCFS non-preemptive processor with deadlines, eqs. (16)-(17)
    if zl, tau = 0; wake = 0; else
      wake = q >= cfree;
      tau = cyc(1)/pf.phi + cyc(2)/pf.phin + wake*pf.twk;
    end
    sig = max(q, cfree); c = sig + tau;
    ok = (c - q) <= dl;
    if ok
      cfree = c; nd(j) = nd(j) + (q >= te); busy = busy + (q >= te)*tau;
      if q >= te && ~zl, Einf = Einf + pf.ecpu*cyc(1) + pf.enpu*cyc(2) + wake*pf.Ewk; end
    else
      drop(j) = drop(j) + (q >= te); c = q;
    end
  end

  function [cand, sel, cyc] = s3_select(f3, kmin3, t3)
    % candidates of S3 for latest item f3 and the items to prefetch:
    % learned S3: candidates with probability >= theta3, at most L;
    % baseline/Markov rule (used by S4 when S3 is the baseline): the
    % N4pf most frequent successors.
    [cand, X3, cnt3] = feat_s3(tr, P, f3, T, cntH, lastIdx, kmin3, t3);
    cyc = pf.cost.scan*Nh/1000;
    if isempty(cand), sel = []; return; end
    if ~strcmp(imp{3}, 'b') && ~strcmp(imp{3}, 'mk')
      p3 = model_eval(WM.(imp{3}), X3); cyc = cyc + cost_of(imp{3}, numel(cand));
      [ps, o] = sort(p3, 'descend'); o = o(ps >= P.theta3); sel = cand(o(1:min(P.L, numel(o))));
    elseif strcmp(imp{3}, 'mk')
      sel = cand(cnt3 >= 2); sel = sel(1:min(P.L, numel(sel)));
    else
      sel = cand(1:min(P.N4pf, numel(cand)));
    end
  end

  function cyc = cost_of(name, n)
    cyc = pf.cost.(name)*n;
  end

  function cyc = s2cycles()
    if learned(2), cyc = cost_of(imp{2}, 1 + P.ne);
    else, cyc = cost_of(base2, max(1, sum(L == 1 | L == 2))); end
  end

  function push(ty, tt, a)
    evT(end+1) = tt; evTy(end+1) = ty; evA{end+1} = a;
  end

  function stage1(tq, kmin, kcur)
    [items, X, ew] = feat_s1(tr, P, kmin, kcur, tq);
    if isempty(items), return; end
    if learned(1)
      [ok, c] = release(1, tq, cost_of(imp{1}, numel(items)), P.T1);
      if ok, sc = model_eval(WM.(imp{1}), X); tq2 = c;
      else, sc = ew; tq2 = tq; end
    else
      [ok, c] = release(1, tq, cost_of('ewma', numel(items)), P.T1);
      sc = ew; tq2 = c; if ~ok, return; end
    end
    [~, o] = sort(sc, 'descend');
    cand = items(o); cand = cand(~(L(cand) == 1 | L(cand) == 2) & ~inflight(cand));
    cand = cand(1:min(P.N1pf, numel(cand)));
    if ~isempty(cand), push(1, tq2, cand(:)'); end
  end

  function stage4(tq, kmin, kcur)
    % S4 predicts which users enter the coverage area within W4; for every
    % predicted user, S3 (model or successor rule) chooses what to prefetch.
    [users, X] = feat_s4(P, ms, tq);
    if isempty(users), return; end
    p = model_eval(WM.(imp{4}), X); us = users(p >= P.theta4);
    idx = kmin:kcur; cand = []; cyc = cost_of(imp{4}, numel(users));
    for u = us(:)'
      ku = idx(tr.u(idx) == u);
      if isempty(ku), continue; end
      [~, s, cy] = s3_select(tr.f(ku(end)), kmin, tq); cyc = cyc + cy;
      cand = [cand; s(:)]; %#ok<AGROW>
    end
    [ok, c] = release(4, tq, cyc, P.T4);
    if ok && ~isempty(cand), push(1, c, unique(cand)'); end
  end

  function candidates(list, tq, kmin)
    % prefetch candidates pass through the admission rule of S2
    for f2 = list
      if L(f2) == 1 || L(f2) == 2 || inflight(f2), continue; end
      [ok, c] = release(2, tq, s2cycles(), tr.db_f(f2));
      if learned(2) && ok
        p = model_eval(WM.(imp{2}), feat_s2(tr, P, f2, lastIdx, kmin, tq, cntH));
        if p < P.theta2, continue; end
        mode = 1;
      else
        mode = 0;
      end
      inflight(f2) = true; push(2, c + tr.db_f(f2), [f2 mode 1]);
      if c >= te, Spf = Spf + tr.s(f2); Npf = Npf + 1; end
    end
  end

  function insert_item(f2, mode, tgt, tq, kmin)
    inflight(f2) = false;
    if L(f2) == 1 || L(f2) == 2 || tr.s(f2) > C, return; end
    if mode == 1
      while used + tr.s(f2) > C
        cached = find(L == 1 | L == 2);
        smp = cached(randi(numel(cached), min(P.ne, numel(cached)), 1));
        sc = model_eval(WM.(imp{2}), feat_s2(tr, P, smp, lastIdx, kmin, tq, cntH));
        [~, m] = min(sc); evict(smp(m), 0);
      end
      setL(f2, 1);
    else
      switch base2
        case 'lru', while used + tr.s(f2) > C, cached = find(L == 1); [~, m] = min(ts(cached)); evict(cached(m), 0); end
        case 'lfu', while used + tr.s(f2) > C
            cached = find(L == 1); sc = fr(cached)*1e9 + ts(cached); [~, m] = min(sc); evict(cached(m), 0); end
        case 'arc', arc_replace(f2, tgt); end
      if strcmp(base2, 'arc') && tgt >= 2, setL(f2, 2); else, setL(f2, 1); end
    end
    ts(f2) = tq; fr(f2) = 1;
    if strcmp(base2, 'arc'), arc_trim(); end
  end

  function setL(f2, v)
    if L(f2) > 0, bL(L(f2)) = bL(L(f2)) - tr.s(f2); end
    if (L(f2) == 1 || L(f2) == 2), used = used - tr.s(f2); end
    L(f2) = v;
    if v > 0, bL(v) = bL(v) + tr.s(f2); end
    if v == 1 || v == 2, used = used + tr.s(f2); end
  end

  function evict(f2, ~)
    if strcmp(base2, 'arc') && ~learned(2)
      setL(f2, 2 + L(f2));        % T1 -> B1 (3), T2 -> B2 (4)
    else
      setL(f2, 0);
    end
  end

  % ---------------- ARC (byte-based) ----------------
  function tgt = arc_ghost(f2)
    tgt = 1;
    if ~strcmp(base2, 'arc'), return; end
    s = tr.s(f2);
    if L(f2) == 3
      parc = min(C, parc + max(1, bL(4)/max(bL(3), 1))*s); setL(f2, 0); tgt = 2;
    elseif L(f2) == 4
      parc = max(0, parc - max(1, bL(3)/max(bL(4), 1))*s); setL(f2, 0); tgt = 3;  % 3 = came from B2
    end
  end

  function arc_replace(f2, tgt)
    while used + tr.s(f2) > C
      T1 = find(L == 1); T2 = find(L == 2);
      if ~isempty(T1) && (bL(1) > parc || (tgt == 3 && bL(1) >= parc) || isempty(T2))
        [~, m] = min(ts(T1)); evict(T1(m), 0);
      else
        [~, m] = min(ts(T2)); evict(T2(m), 0);
      end
    end
  end

  function arc_trim()
    while bL(1) + bL(3) > C && bL(3) > 0
      B1 = find(L == 3); [~, m] = min(ts(B1)); setL(B1(m), 0);
    end
    while sum(bL) > 2*C && bL(4) > 0
      B2 = find(L == 4); [~, m] = min(ts(B2)); setL(B2(m), 0);
    end
  end
end
