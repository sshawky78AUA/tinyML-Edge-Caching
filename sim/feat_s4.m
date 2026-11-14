function [users, X] = feat_s4(P, mobstate, t)
% FEAT_S4  Features of stage S4 at time t for the users currently in the
% two cells adjacent to the EN cell (cells 2 and Nc on the ring).
% mobstate: struct with per-user fields cell, tenter, hist (last Q4 cells),
% and visits1 (number of past entries into cell 1).
c = mobstate.cell;
users = find(c == 2 | c == P.Nc);
n = numel(users);
X = zeros(n, 4 + P.Q4 + 2);
for a = 1:n
  u = users(a);
  h = mobstate.hist(u,:);                         % last Q4 cells, most recent first
  rel = arrayfun(@(x) ring_rel(x, P.Nc), h)/(P.Nc/2);
  ts = 1; tr0 = 0; if isfield(P, 'tscale'), ts = P.tscale; tr0 = P.tref; end
  tt = tr0 + (t - tr0)*ts;
  X(a,:) = [c(u) == P.Nc, c(u) == 2, log1p((t - mobstate.tenter(u))*ts), ...
            log1p(mobstate.visits1(u)), rel, sin(2*pi*tt/86400), cos(2*pi*tt/86400)];
end
end

function r = ring_rel(x, Nc)
% signed ring distance of cell x from cell 1; 0 if unknown (x == 0)
if x == 0, r = 0; return; end
r = mod(x - 1 + Nc/2, Nc) - Nc/2;
end
