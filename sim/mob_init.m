function ms = mob_init(tr, P, t)
% MOB_INIT  Mobility state of all users at time t (entries of the log up to t).
ms.cell = zeros(tr.U,1); ms.tenter = zeros(tr.U,1);
ms.hist = zeros(tr.U, P.Q4); ms.visits1 = zeros(tr.U,1); ms.ptr = 1;
ms = mob_advance(ms, tr, P, t);
end
