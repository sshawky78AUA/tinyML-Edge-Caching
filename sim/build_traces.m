function build_traces()
% BUILD_TRACES  Build the two request traces of the paper.
%  data/trace_s1p.mat : synthetic trace (Section VI-B), generator seed 1
%  data/trace_mt.mat  : MovieTweetings 2017 window, from data/mt2017.csv
% The parameters of the synthetic trace are in params_default.m.
% In GNU Octave both files are identical to the ones used in the paper.
% MATLAB uses a different random number generator for rand('state', .),
% so in MATLAB the synthetic trace has the same statistics but different
% individual requests; the MovieTweetings trace is identical in both.
P = params_default();
tr = gen_trace(P, 1);
tr = prep_trace(tr, P);
save('../data/trace_s1p.mat', 'tr', '-v7');
fprintf('synthetic: K=%d U=%d F=%d T=%.0f s\n', tr.K, tr.U, tr.F, P.T);
build_mt();
end
