function y = model_eval(W, X)
% MODEL_EVAL  Forward pass of an exported model on the rows of X.
% Dense models return probabilities (n x 1); embedding models (S3) take
% integer token rows (1-based, as produced by feat_s3) and return logits.
if strcmp(W.kind, 'emb')
  n = size(X,1); d = size(W.E,2); Z = zeros(n, size(X,2)*d);
  for q = 1:size(X,2)
    Z(:, (q-1)*d+1:q*d) = W.E(X(:,q)+1, :);        % Keras index v -> MATLAB row v+1
  end
  A = Z;
else
  A = (X - W.mu) ./ W.sd;
end
for l = 1:W.nlayers
  A = A * W.(sprintf('W%d', l)) + W.(sprintf('b%d', l));
  if l < W.nlayers, A = max(A, 0); end
end
if strcmp(W.kind, 'emb'), y = A; else, y = 1./(1 + exp(-A)); end
end
