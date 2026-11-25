function build_mt()
% BUILD_MT  Request trace from the MovieTweetings ratings of 2017 [Dooms et
% al. 2013]: every rating is a request of its user for the movie. Arrival
% times are scaled by gamma so that the mean request rate equals that of the
% synthetic trace; all movies have the same size (the dataset has no sizes);
% the dataset has no user locations, so S4 is not evaluated on this trace.
P = params_default();
a = csvread('../data/mt2017.csv');
tr.t = a(:,1); tr.u = a(:,2); tr.f = a(:,3);
tr.K = numel(tr.t); tr.U = max(tr.u); tr.F = max(tr.f);
tr.s = 20e3*ones(tr.F, 1); tr.mob = zeros(0, 3);
P.T = ceil(max(tr.t));
tr = prep_trace(tr, P); tr.name = 'mt2017';
save('../data/trace_mt.mat', 'tr', '-v7');
fprintf('K=%d U=%d F=%d T=%.0f s Ttr=%.0f Tva=%.0f\n', tr.K, tr.U, tr.F, P.T, tr.Ttr, tr.Tva);
end
