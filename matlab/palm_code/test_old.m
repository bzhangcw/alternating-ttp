clear;
addpath('data');
rng('default');
% model = gurobi_read('data/test_15_5_60.mps');
% filename = 'data/index_dict_15_5_60.pkl';
model = gurobi_read('data/test_20_29_300.mps');
filename = 'data/index_dict_20_29_300.pkl';
% model = gurobi_read('data/test_50_29_300.mps');
% filename = 'data/index_dict_50_29_300.pkl';
params.outputflag = 0;
params.mipgap = 0.0001;
params.TimeLimit = 300;
params.itermax = 250;

[m, n] = size(model.A);
model.obj = -rand(n, 1);
model.vtype = char('C'*ones(size(model.vtype)));
rgrb = gurobi(model);
[subproblems, coupling] = initialization(filename, model);
    

x = palm(subproblems, coupling, model);
% x = ipalm(subproblems, coupling, model);
%palm_l(subproblems, coupling, model);
