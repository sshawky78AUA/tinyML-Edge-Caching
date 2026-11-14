function pf = platforms(P, cyc, vela, mdl)
% PLATFORMS  Profiles of the four platform classes and the per-inference
% cost of every implementation. Sources (see the paper, Table IV):
%  C1 STM32F103RB  datasheet DS5319: 27 mA run (72 MHz, flash, periph. off),
%                  14 uA Stop (LP regulator), 5.4 us wake-up; VDD = 3.3 V.
%  C2 STM32F407VG  datasheet DS8626: 40 mA run (168 MHz, ART on, periph. off),
%                  0.28 mA Stop (LP regulator, flash deep power-down), 110 us wake-up.
%  C3 Alif E3 HE   datasheet: 29 uA/MHz dynamic (M55-HE, 160 MHz); vendor
%                  measurements: 25 uA standby, 4 us wake-up (3.38 V).
%                  Ethos-U55-128 cycles from Vela (Deep_Embedded, Sram_Only).
%                  e_npu = e_cpu is DERIVED (vendor: 78x faster, 76x less energy).
%  C4 GAP8         PULP-NN on GAP8 at 170 MHz, 1.2 V: 1.07 GMAC/s, 0.27 mJ per
%                  frame at 241 fps -> 6.3 MAC/cycle, 64.8 mW (DERIVED);
%                  30 uW state-retentive sleep [Flamand 2018]; wake-up not
%                  published (set to 0, optimistic).
% cyc: struct of modelled Cortex-M cycles (fw/cycles.json); vela: struct of
% Ethos-U55-128 cycles per model; mdl: struct with macs and tflite bytes.
KB = 1024;
names = {'s1_lin','s1_mlp_s','s1_mlp_l','s2_lin','s2_mlp_s','s2_mlp_l', ...
         's3_lin','s3_mlp_s','s3_mlp_l','s4_lin','s4_mlp_s','s4_mlp_l'};

% ---- Class 1 ----
p.name = 'C1'; p.phi = 72e6; p.phin = 72e6; V = 3.3;
p.ecpu = V*27e-3/p.phi; p.enpu = p.ecpu; p.Pslp = V*14e-6;
p.twk = 5.4e-6; p.Ewk = V*27e-3*p.twk;
p.Mram = 20*KB; p.Mfl = 128*KB; p.label = 'modelled cycles (M3 TRM timings)';
p.cost = core_costs(cyc, 'm3', names, 0); pf{1} = p;

% ---- Class 2 ----
p.name = 'C2'; p.phi = 168e6; p.phin = 168e6;
p.ecpu = V*40e-3/p.phi; p.enpu = p.ecpu; p.Pslp = V*0.28e-3;
p.twk = 110e-6; p.Ewk = V*40e-3*p.twk;
p.Mram = 192*KB; p.Mfl = 1024*KB; p.label = 'modelled cycles (M4 TRM timings)';
p.cost = core_costs(cyc, 'm4', names, 0); pf{2} = p;

% ---- Class 3: Alif Ensemble E3, high-efficiency subsystem ----
p.name = 'C3'; p.phi = 160e6; p.phin = 160e6;
p.ecpu = 3.3*29e-6/1e6; p.enpu = p.ecpu; p.Pslp = 3.38*25e-6;
p.twk = 4e-6; p.Ewk = 3.3*29e-6*160*p.twk;
p.Mram = 512*KB; p.Mfl = 5.5*1024*KB; p.label = 'Vela (NPU) + M4 model (CPU)';
p.cost = core_costs(cyc, 'm4', names, 0);           % baselines and CPU parts
for a = 1:numel(names)
  n = names{a}; cpu = 0;
  p.cost.(n) = [cpu, vela.(n)];
end
pf{3} = p;

% ---- Class 4: GAP8 ----
p.name = 'C4'; p.phi = 170e6; p.phin = 170e6;
Pact = 0.27e-3*241;                                  % W at the PULP-NN operating point
p.ecpu = Pact/p.phi; p.enpu = Pact/p.phin; p.Pslp = 30e-6;
p.twk = 0; p.Ewk = 0;
p.Mram = 512*KB; p.Mfl = 8*1024*KB; p.label = 'modelled (PULP-NN throughput)';
p.cost = core_costs(cyc, 'm3', names, 0);           % fabric controller ~ single-issue core
r = 1.07e9/170e6;                                    % MAC per cluster cycle
for a = 1:numel(names)
  n = names{a}; cpu = 0;
  p.cost.(n) = [cpu, ceil(mdl.(n).macs / r)];
end
pf{4} = p;

% common entries
for c = 1:4
  pf{c}.eta = 0;                   % whole RAM retained in the chosen sleep mode
  pf{c}.Mram0_base = 4*KB;         % ASSUMED firmware RAM
  pf{c}.Mfl0 = 32*KB;              % ASSUMED firmware flash
  for a = 1:numel(names)
    n = names{a};
    pf{c}.mufl.(n) = mdl.(n).bytes;
    pf{c}.muram.(n) = mdl.(n).ram;
  end
end
end

function cost = core_costs(cyc, core, names, ~)
e = cyc.([core '_empty']);
for a = 1:numel(names)
  cost.(names{a}) = [cyc.([core '_' names{a}]) - e, 0];
end
cost.ewma = [cyc.([core '_ewma_item']) - e, 0];                 % per tracked item
cost.lru = [(cyc.([core '_lru_scan1000']) - e)/1000, 0];       % per cached item
cost.lfu = [(cyc.([core '_lfu_scan1000']) - e)/1000, 0];
cost.arc = cost.lru;
cost.scan = [cyc.([core '_lfu_scan1000']) - e, 0];               % per 1000 history entries (S3 candidates)
% S3 scores each candidate with a dense model of 6 inputs, the same
% architecture as the S2 model of the same size
for v = {'lin', 'mlp_s', 'mlp_l'}
  cost.(['s3_' v{1}]) = cost.(['s2_' v{1}]);
end
end
