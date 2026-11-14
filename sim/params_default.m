function P = params_default()
% PARAMS_DEFAULT  Nominal parameters of the pilot study (synthetic traces).
% Values marked ASSUMED are pilot assumptions, not taken from a source.

% ---- synthetic trace ----
P.F = 5000;            % catalogue size
P.U = 200;             % users
P.Nc = 8;              % cells on a ring; cell 1 = EN coverage area
P.T = 2*86400;         % trace duration (s)
P.ru = 0.02;           % request rate of a user inside the EN cell (1/s)
P.dwell_mean = 1800;   % mean dwell time in a cell (s)
P.dwell_shape = 3;     % gamma shape of the dwell time
P.persist = 0.8;       % prob. of keeping the direction of movement
P.alpha = 0.8;         % Zipf exponent
P.beta = 0.35;         % prob. that a request follows the user's previous item
P.succ_p = [0.6 0.3 0.1];
P.Tdrift = 6*3600;     % popularity reshuffle period (s)
P.drift_top = 500; P.drift_p = 0.3;
P.size_median = 20e3; P.size_sigma = 1.0; P.size_min = 1e3; P.size_max = 1e6;

% ---- split (fractions of duration) ----
P.frac_tr = 0.5; P.frac_va = 0.2;

% ---- links (ASSUMED pilot values) ----
P.da = 0.005; P.Ra = 1.25e6;    % 5 ms, 10 Mb/s
P.db = 0.050; P.Rb = 0.25e6;    % 50 ms, 2 Mb/s
P.ea = 0.1e-6; P.eb = 1.0e-6;   % J per byte

% ---- cache ----
P.Cfrac = 0.01;        % capacity as fraction of unique test bytes
P.bx = 8;              % index bytes per cached item
P.br = 8;              % bytes per stored request in the history
P.Nh_inf = 20000;      % longest history used in the study

% ---- stages ----
P.T1 = 600; P.nw = 6; P.W1 = 600; P.N1pf = 5; P.rho_ewma = 0.5;
P.W2 = 3600; P.ne = 16; P.theta2 = 0.3; P.nd = 4;
P.Q3 = 3; P.Fe = 512; P.L = 2; P.Nc3 = 8; P.theta3 = 0.15;  % theta3 selected on validation (best F1)
P.T4 = 300; P.W4 = 600; P.Q4 = 3; P.N4pf = 2; P.theta4 = 0.2;  % selected on validation (best F1)
P.Tw = 3600;           % window for H_l
P.tscale = 1; P.tref = 0;   % rate sweep: stages measure time in trace units, t_trace = tref + (t - tref)*tscale
end
