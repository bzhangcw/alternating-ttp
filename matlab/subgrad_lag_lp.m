%% MILP examples 
% (c) Chuwen Zhang, chuwen@shanshu.ai.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this example illustrate the case where subgradient method
%   won't produce primal feasible solution for Lagrangian relaxation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
% [grbm, model] = read_model('test_15_10_120.lp');
[grbm, model] = read_model('test_12_5_80.lp');

%% Sanity check
rgrb = gurobi(grbm);

%% PDGH on Lagrangian
% 
% select D to do the relaxation.
% let, 
%   [B; D] = A;
% use the same rule for rhs.
% consider:
%  min_l max_x c'x - l'Dx + l'd 
%   s.t.   Bx <= b
% also translate to minimization
c = model.c; D = model.D; d = model.d;
[m, n] = size(model.B);
[k, n] = size(D);
% unchanged part of subprob of x

subproblem.A = model.B;
subproblem.rhs = model.b;
subproblem.lb = model.lb;
subproblem.ub = model.ub;
subproblem.vtype = 'C';
subproblem.sense = grbm.sense(1:m);
subproblem.modelsense = 'max';

lk = ones(k, 1);
lb = lk;
xk = zeros(n, 1);
xm = zeros(n, 1);
xp = 0;
% step-sizes
alp = 1.0;
eta = 0.01;
step = 0;
maxiter = 1200;
ct_unimprov = 0;
bprim = 0;
bdual = 1e6;
bfeas = 0;
bfeasx = 0;
pars.OutputFlag = 0;
% compute objective
disp(" k      c'x      phi    |Dx - d|  |Dxb - d| |x-xk|      eta     alpha    step");
for i=1:maxiter
  % solve x
  obj = c - D'*lk;
  subproblem.obj = obj;
  r = gurobi(subproblem, pars);
  xk = r.x;
  phik = r.objval + lk'*d;
  zk = c'*xk;
  fprintf("%.3d %.2e %.2e %.3e %.3e %.3e %.2e %.2e %.2e\n", ...
    i, bfeas, ...
    phik, ...
    abs(min(d - D*xk)), abs(min(d - D*xm)), ...
    norm(xk - xp), ...
    eta, alp, step ...
  );
  
  if min(d - D*xk) >= 0
    if zk > bfeas
      bfeas = zk;
      bfeasx = xk;
    end
  end

  if phik < bdual
    bdual = phik;
    lb = lk;
    ct_unimprov = 0;
  else
    ct_unimprov = ct_unimprov + 1;
  end
  
  if ct_unimprov > 45
    alp = alp * 0.5;
    ct_unimprov = 0;
  end
 
  xm = xm * 0.98 + xk * 0.02;
  pf = d - D * xm;
  eta = (bdual - bprim) / (norm(pf)^2);
  step = eta * alp;
  if (step < 1e-6)
    break
  end
  lk = max(lk - step * pf, 0);
  xp = xk;
  
end

fprintf("%.3d %.2e %.2e %.3e %.3e %.3e %.2e %.2e %.2e\n", ...
    i, bfeas, ...
    phik, ...
    abs(min(d - D*xk)), ...
    abs(min(d - D*xm)), ...
    norm(xk - r.x), ...
    eta, alp, step ...
);
