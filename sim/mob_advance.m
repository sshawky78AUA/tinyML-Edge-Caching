function ms = mob_advance(ms, tr, P, t)
% MOB_ADVANCE  Apply all mobility log entries with time <= t.
n = numel(tr.mobT);
while ms.ptr <= n && tr.mobT(ms.ptr) <= t
  u = tr.mobU(ms.ptr); c = tr.mobC(ms.ptr);
  if ms.cell(u) > 0, ms.hist(u,:) = [ms.cell(u), ms.hist(u,1:end-1)]; end
  ms.cell(u) = c; ms.tenter(u) = tr.mobT(ms.ptr);
  if c == 1, ms.visits1(u) = ms.visits1(u) + 1; end
  ms.ptr = ms.ptr + 1;
end
end
