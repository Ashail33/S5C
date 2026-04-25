% Selective sampling-based scalable sparse subspace clustering.
% Param:
%   X: data matrix (each column is a data point)
%
%   lambda: sparsity regularization hyperparameter
%
%   num_subsamples: number of subsamples
%
%   reltol: relative tolerance
%
%   num_I_t: number of samples to approximate gradients for each candidate
%
%   seed: random seed
%
%   verbose: verbosity
%
% Return:
%
%   C: representation matrix learned by SSC with subsamples
%
%   S_t: data id (column number) of subsamples selected by S5C
%
%  [Cite] Shin Matsushima and Maria Brbic,
%         "Selective Sampling-based Scalable Sparse Subspace Clustering"
%         Conference on Neural Information Processing Systems, 2019
%
%  version 1.0 -- Oct/2019
%
%  Shin Matsushima and Maria Brbic


function [C, S_t] = representation_learning_S5C( X, lambda, num_subsamples, ...
						 reltol, num_I_t, seed, verbose)
%% parameter spesification
if ~exist('reltol','var')
    reltol = 10^-3;
end
if ~exist('num_I_t','var')
    num_I_t = 1;
end
if ~exist('seed','var')
  rng(1234)
else
  rng(seed)
end
if ~exist('verbose','var')
   verbose = false;
end

[m,n] = size(X);
if( n < num_subsamples)
  warning = ['number of subsamples are set to be larger than the number of ' ...
             'datapoints. automatically fixed it to be number of datapoints']
  num_subsamples = n;
end

% iteration is taken to be long enough
max_t = num_subsamples*2;
num_rand_idx = num_subsamples*2*num_I_t;
if n >= num_rand_idx
    rand_idx = randperm(n, num_rand_idx);
else
    rand_idx=[];
    for it = 1:ceil( num_rand_idx / n )
        rand_idx = [rand_idx, randperm(n)];
    end
end

S_t = zeros(1, num_subsamples); % preallocated; truncated after the loop
num_St = 0;

norms = ones(1,size(X,1)) * (X.^2);
stats = struct (...
    'reltol', reltol,...
    'Lambda', lambda,...
    'time_for_CD',0,...
    'time_for_fit',0,...
    'normsSt', zeros(1, num_subsamples) ,...
    'XSt', zeros(m, num_subsamples) ,...
    'W', zeros(num_subsamples,n),... % C_St
    'R', X ) ; % R = X - XC

%% select selective subsamples. 
selectiontic = tic;
for t = 1:max_t
  
    I_t = rand_idx((t-1)*num_I_t+1:t*num_I_t);
    
    if(num_St ~= 0)
      S_t_active = S_t(1:num_St);
      for i=I_t
        stats = mylasso(X, S_t_active, stats, i);
      end
    end

    % Soft-thresholded gradient over all columns at once. Then mask out
    % indices already in S_t and the queried indices I_t so they cannot be
    % selected.
    grad_L = (stats.R(:,I_t))' * X;
    grad = min(grad_L+lambda, max(0, grad_L-lambda));
    if(num_St ~= 0)
      grad(:, S_t(1:num_St)) = 0;
    end
    grad(:, I_t) = 0;

    [max_val, dSt] = max(sum(grad.^2,1));

    if( max_val ~= 0)
      num_St = num_St + 1;
      S_t(num_St) = dSt;     % dSt is i' in the paper
      stats.normsSt(num_St) = norms(dSt);
      stats.XSt(:, num_St) = X(:, dSt);
      if(num_St == num_subsamples)
	break
      end
    end
end
% trim preallocated buffers to the number of samples actually selected
S_t = S_t(1:num_St);
stats.normsSt = stats.normsSt(1:num_St);
stats.XSt = stats.XSt(:, 1:num_St);
selectiontime = toc(selectiontic);

%% solve all lasso with final S_t
lassotic = tic;
for i=1:n
 stats = mylasso(X, S_t, stats, i);
end
lassotime = toc(lassotic);

%% put C into sparse matrix format
settic = tic;
row_idx = repmat( S_t, 1, n);
col_idx = repmat( 1:n, num_St, 1);
C = sparse( row_idx, col_idx, stats.W(1:num_St,:), n, n);
setctime = toc(settic);

%% show stats
if verbose
  timestats = containers.Map();
  timestats('1.selection') = selectiontime;
  timestats('2.final lasso') = lassotime;
  timestats('3.fit in all lasso') = stats.time_for_fit;
  timestats('4.CD in fit') = stats.time_for_CD;
  timestats('5.set C') = setctime;

  timestats = [keys(timestats) ; values(timestats)]
end

%obj = 0.5 * norm(X-X*C, 'fro')^2 + lambda * sum(sum(abs(C)));

end
