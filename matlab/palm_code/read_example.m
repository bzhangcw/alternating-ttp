clear;
addpath('data');
rng('default');
filename = 'data/ttp_25_29_300.mat';
load(filename, 'trains', 'b');

%%
check = 1;

%
ntotal = length(trains);
nn = 1;
vars = 0;
for k = trains
  % we do not need coupling like before
  [m, nvar] = size(k{1}.A);
  subproblems(nn).coupling = k{1}.A;
  subproblems(nn).A = k{1}.B;
  subproblems(nn).obj = k{1}.c;
  subproblems(nn).sense = k{1}.sense_B_k;
  subproblems(nn).c = k{1}.c;
  subproblems(nn).rhs = k{1}.b;
%   subp.lb = 0;
%   subp.ub = 1;
  subproblems(nn).vtype = 'B';
  subproblems(nn).lb = zeros(nvar,1);
  subproblems(nn).ub = ones(nvar,1);
  subproblems(nn).vars_index = vars+1: vars+nvar;
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




