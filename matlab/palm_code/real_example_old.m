clear;
addpath('data')
addpath('alm')
name='20_29_300';
mps=sprintf('data/test_%s.mps', name);
pkl=sprintf('data/index_dict_%s.pkl', name);

model = gurobi_read(mps);
filename = pkl;
params.outputflag = 0;
params.mipgap = 0.0001;
params.TimeLimit = 300;
params.itermax = 250;

% relax or not
% model.vtype = char('C'*ones(size(model.vtype)))
rgrb = gurobi(model)

[subproblems, coupling] = initialization(filename, model);
    
% create algorithm struct
alg = init_alg_struct(model, coupling, params);

save(sprintf(name))