function [Nh, feas, M0] = nh_for(impl, pf, P, C, smean)
% NH_FOR  History length allowed by Definition 1 for a configuration on a
% platform: the largest N_h with N_h*b_r + sum(mu_ram) <= M_ram - M_ram0, and
% the flash condition sum(mu_fl) <= M_fl - M_fl0. The reserved RAM includes
% the cache index, n_x*b_x with n_x = ceil(2C/s_mean) entries (ASSUMED sizing).
nx = ceil(2*C/smean);
M0 = pf.Mram0_base + nx*P.bx;
ram = 0; fl = 0;
for j = 1:4
  if ~any(strcmp(impl{j}, {'b','mk'})), ram = ram + pf.muram.(impl{j}); fl = fl + pf.mufl.(impl{j}); end
end
feas = fl <= pf.Mfl - pf.Mfl0;
Nh = min(P.Nh_inf, floor((pf.Mram - M0 - ram)/P.br));
if Nh < P.Q3 || ~feas, feas = false; Nh = 0; end
end
