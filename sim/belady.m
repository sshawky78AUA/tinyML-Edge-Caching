function H = belady(tr, C, seg)
% BELADY  Hit ratio of Belady's rule (evict the item whose next request is
% furthest in the future) on [te, t1), with immediate insertion of every
% missed item. For items of different sizes this is a reference, not a bound.
t0 = seg(1); te = seg(2); t1 = seg(3);
inC = false(tr.F,1); nextReq = inf(tr.F,1); used = 0; hit = [];
ks = find(tr.t >= t0 & tr.t < t1)';
for k = ks
  f = tr.f(k); nk = tr.nextIdx(k);
  nr = inf; if nk <= tr.K, nr = tr.t(nk); end
  if inC(f)
    if tr.t(k) >= te, hit(end+1) = 1; end %#ok<AGROW>
  else
    if tr.t(k) >= te, hit(end+1) = 0; end %#ok<AGROW>
    if tr.s(f) <= C && nr < inf
      while used + tr.s(f) > C
        cached = find(inC); [far, m] = max(nextReq(cached));
        if far <= nr, break; end
        inC(cached(m)) = false; used = used - tr.s(cached(m));
      end
      if used + tr.s(f) <= C, inC(f) = true; used = used + tr.s(f); end
    end
  end
  nextReq(f) = nr;
end
H = mean(hit);
end
