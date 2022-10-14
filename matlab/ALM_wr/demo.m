clear;
%addpath('/Users/wangrui/Desktop/code_mat/data');
addpath(genpath(pwd));
rng('default');
%filename = 'data/ttp_100_29_720.mat';
filename = 'data/ttp_50_29_300_2.mat';
%filename = 'ttp_292_29_1080.mat';
%filename = 'data/ttp_200_29_720.mat';
%filename = 'data/ttp_150_29_1080.mat';
%filename = 'data/ttp_120_29_400.mat';
%filename = 'data/ttp_25_29_300.mat';
%filename = 'data/ttp_examples/ttp_10_10_200.mat';
%filename = 'data/ttp_examples/ttp_10_10_300.mat';
%filename = 'data/ttp_examples/ttp_20_10_200.mat';
%filename = 'data/ttp_examples/ttp_20_10_300.mat';
%filename = 'data/ttp_examples/ttp_50_10_200.mat';
%filename = 'data/ttp_examples/ttp_50_10_300.mat';
%filename = 'data/ttp_examples/ttp_50_15_600.mat';
%filename = 'data/ttp_examples/ttp_100_15_600.mat';
%filename = 'data/ttp_examples/ttp_100_20_720.mat';
load(filename, 'trains', 'b');

%%
check = 1;

%
ntotal = length(trains);
nn = 1;
vars = 0;
%trains{1}.c(randperm(length(trains{1}.c),10))=-1;
for k = trains
  
  % we do not need coupling like before
  [m, nvar] = size(k{1}.A);
  subproblem(nn).coupling = k{1}.A;
  subproblem(nn).A = k{1}.B;
  subproblem(nn).obj = k{1}.c;
  subproblem(nn).sense = k{1}.sense_B_k;
  subproblem(nn).c = k{1}.c;
  subproblem(nn).rhs = k{1}.b;
%   subp.lb = 0;
%   subp.ub = 1;
  subproblem(nn).vtype = 'B';
  subproblem(nn).lb = zeros(nvar,1);
  subproblem(nn).ub = ones(nvar,1);
  subproblem(nn).vars_index = vars+1: vars+nvar;
  %% for gurobi
  A{nn} = k{1}.A;
  B{nn} = k{1}.B;
  c{nn} = k{1}.c;
  bc{nn} = k{1}.b;
  sense{nn} = k{1}.sense_B_k;
  nn = nn + 1;
  vars = vars + nvar;
end

nonbinding = blkdiag(B{:});
binding = [A{:}];
coupling.A = binding;
coupling.rhs = b;

%% for gurobi
model.A = [nonbinding;binding];
model.rhs = [vertcat(bc{:}); b];
model.sense = [vertcat(sense{:}); trains{1,1}.sense_A_k;];
model.obj = [vertcat(c{:})];
model.vtype = 'B';

% tic;
% resultg=gurobi(model);
% gtime=toc;

Funcollect=EvaluationFunc1;



%% ALM-proxl
% tic;
% [x_l] = Funcollect.palm_l(subproblem,coupling,model);
% t_almp = toc;
%% ALM-original
tic;
[x,k] = Funcollect.palm_dual(subproblem,coupling,model);
t_almo = toc;

