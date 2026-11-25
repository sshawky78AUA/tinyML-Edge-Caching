function export_all()
% EXPORT_ALL  Write the training and validation examples of every stage.
%  data/train_syn/{tr,va}_s{1..4}.csv, {tr,va}_s3_req.csv  (synthetic trace)
%  data/train_mt/{tr,va}_s{1..4}.csv,  {tr,va}_s3_req.csv  (MovieTweetings)
% Each row holds the features of one example followed by its label.
% Training examples start one hour into the synthetic trace (S1, S2, S4) or
% at its start (S3, and every stage of MovieTweetings), as in the paper.
P = params_default();
d = '../data/train_syn/'; if ~exist(d, 'dir'), mkdir(d); end
S_ = load('../data/trace_s1p.mat'); tr = S_.tr;
export_training(tr, P, 3600, tr.Ttr, [d 'tr']);
export_training(tr, P, tr.Ttr, tr.Tva, [d 'va']);
export_s3c([d 'tr'], 0, tr.Ttr, '../data/trace_s1p.mat');
export_s3c([d 'va'], tr.Ttr, tr.Tva, '../data/trace_s1p.mat');
d = '../data/train_mt/'; if ~exist(d, 'dir'), mkdir(d); end
S_ = load('../data/trace_mt.mat'); tr = S_.tr; P.T = ceil(max(tr.t));
export_training(tr, P, 0, tr.Ttr, [d 'tr']);
export_training(tr, P, tr.Ttr, tr.Tva, [d 'va']);
export_s3c([d 'tr'], 0, tr.Ttr, '../data/trace_mt.mat');
export_s3c([d 'va'], tr.Ttr, tr.Tva, '../data/trace_mt.mat');
end
