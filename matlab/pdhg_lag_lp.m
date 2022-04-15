%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this example test if PDGH works at Lagrangian relaxation.
% (c) Chuwen Zhang.
% @date: 04/06/2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[grbm, model] = read_model('test_10_5_50.lp');

%% Sanity check
rgrb = gurobi(grbm);

%% PDGH on Lagrangian
% 
% select D to do the relaxation.
% let, 
%   [B; D] = A;
% use the same rule for rhs.
% consider:
%   min_x max_l c'x + l'Dx - l'd 
%   s.t.   Bx <= b
% also translate to minimization
c = - model.c; D = model.D; d = model.d;
[m, n] = size(grbm.A);
[k, n] = size(D);
% unchanged part of subprob of x

subproblem.A = model.B;
subproblem.rhs = model.b;
subproblem.lb = model.lb;
subproblem.ub = model.ub;
subproblem.vtype = 'C';
subproblem.sense = '<';
subproblem.modelsense = 'min';

lk = ones(k, 1);
xk = zeros(n, 1);

dnom = normest(D);
% step-sizes
% eta <= (2|D|^2 *tau)^{-1}
eta = 0.1/dnom;
tau = 0.9/2/dnom^2/eta;

logiter = 1;
maxiter = 1000;
pars.OutputFlag = 0;
% compute objective
disp(" k     c'x    |Dx - d|   |x-xk|");
for i=1:maxiter
  
  % solve x
  obj = c + D'*lk - xk/tau;
  qmat = speye(n)/2 / tau;
  subproblem.Q = qmat;
  subproblem.obj = obj;
  r = gurobi(subproblem, pars);
 
  
  info = sprintf("%.3d %.3e %.3e %.3e\n", i, c'*xk, norm(D*xk - d), norm(xk - r.x));
  if mod(i, logiter) == 0
    fprintf(info)
  end
  lk = max(lk + eta * D * (2*r.x - xk) - eta * d, 0);
  xk = r.x;
%  eta = eta * 1.05;
%  tau = tau / 1.05;
  
end

fprintf("ground truth, obj := %.4e\n", rgrb.objval);
disp("pdhg info:")
disp(" k     c'x    |Dx - d|   |x-xk|");
fprintf(info);
  

