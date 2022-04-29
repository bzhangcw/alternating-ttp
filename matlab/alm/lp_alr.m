% For LP (and many convex problems), LR+ gives you primal feas.
%   ALR goes to KKT points.
% we construct an example,
%
% z = min c'x
%     s.t Bx = q
%         x \in [l, u]
% then relax rows by `index` to solve the dual problem
% phi = min c'x + l'Dx - l'd
%     s.t Ax = b;
%         x \in [l, u]
% where, B = [D; A], q = [d; b]
% phi(l) is called the dual function of l
function [info] = lp_alr(model, index, pars)

[m, n] = size(model.A);
dim = sum(index);
% unchanged part of subprob of x
D = [model.A(index, :) speye(dim)];
d = model.rhs(index, :);
c = [model.obj; sparse(dim, 1)];

subproblem.A = [model.A(~index, :) sparse(m - dim, dim)];
subproblem.rhs = model.rhs(~index, :);
subproblem.lb = [model.lb; zeros(dim, 1)];
subproblem.ub = [model.ub; ones(dim, 1)];
subproblem.vtype = [model.vtype; char(67*ones(dim, 1))];
subproblem.sense = model.sense(~index,:);

lk = rand(dim, 1);
xk = zeros(n, 1);
Q = D'*D;
% step-sizes
rho = pars.rho;
pfeasb = 0;
headers = ["c'x", "phi", "|Dx - d|", "gap", "rho"];
slots = ["%10s", "%10s", "%13s", "%8s", "%10s"];
header = 'k';
for j=1:5
  header=strcat(header, sprintf(slots(j), headers(j)));
end
header = strcat(header, '\n');

for i=1:pars.itermax
  % 
  if ~mod(i - 1, 50)
    fprintf(header);
  end
  % solve x
  obj = full(c + D'*lk - rho*D'*d);
  subproblem.Q = Q*rho/2;
  subproblem.obj = obj;
  subproblem.objcon = - lk'*d + rho/2*(d'*d);
  r = gurobi(subproblem, pars);
  % collect iteration
  xk = r.x;
  % xm = xm*0.98*(i>1) + xk*0.02^(i>1);
  phik = r.objval;
  zk = c'*xk;
  gapk = 100*abs(zk-phik)/(abs(phik)+1e-6);
  psub = D*xk - d;
  pfeas = norm(psub);
  fprintf("%+.3d %+.2e %+.2e %+.3e %+.1e%% %+.3e\n", ...
    i, ...
    zk, ...
    phik, ...
    pfeas, ...
    gapk, ...
    rho ...
  );
  if norm(psub) < 1e-6
    break
  end
  %% update rho
  if pfeas > pfeasb
    rho = 1.2 * rho;
  end
  lk = lk + rho*psub;
  pfeasb = pfeas;
end
fprintf("%+.3d %+.2e %+.2e %+.3e %+.1e%% %+.3e\n", ...
    i, ...
    zk, ...
    phik, ...
    norm(psub), ...
    gapk, ...
    rho ...
  );
info.k = i;
info.x = xk(1:n);
info.l = lk;
info.rho = rho;
info.z = zk;
info.phik = phik;
info.gap = gapk;
info.D = D;
info.d = d;
info.pfeas = norm(psub);
end
