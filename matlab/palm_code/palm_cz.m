% Model
% min c'x s.t. Ax<=b, Bx<=d, x \in {0,1}
% min c'x+rho*\|max{Ax-b+lambda/rho,0}\|^2 s.t. Bx<=d, x \in {0,1}
% indefinite proximal version,
% includes an extrapolation step.
function [x] = palm_cz(subproblem,coupling,model)
A  = coupling.A;
b  = coupling.rhs;

eps    = 1e-6;
[m,n]  = size(A);
x      = ones(n,1);
rho    = 1e-2;
lambda = rho*ones(m,1);
nblks  = length(subproblem);
for j = 1:nblks
  importance(j) = subproblem.binding;
end
[~, I] = sort(importance,'descend');
sigma  = 2;
% [~,y]  = eigs(A'*A);
% tau    = 1/(max(diag(y))*rho);
%
Anorm = 20 % normest(A);
tau    = 1/(Anorm^2*rho);
kmax   = 1000;
imax   = 10;
% subproblem = values(subproblems);

headers = ["c'x", "lobj", "|Ax - b|", "error", "rho","tau","iter"];
slots = ["%10s", "%10s", "%14s", "%8s", "%10s", "%9s","%9s"];
header = 'k';
for j=1:7
  header=strcat(header, sprintf(slots(j), headers(j)));
end
header = strcat(header, '\n');
fprintf(header);
for k = 1 : kmax
  xk=x;
  for iter = 1 : imax
    x_old = x;
    for j = 1:nblks % cyclic
    % for j = 1:randperm(nblks) % randomized
    % for j = [1:nblks nblks-1:-1:1] % double-sweep
    % for j = I % sort by num of importance
      cj  = subproblem(j).c;
      ATj = subproblem(j).coupling;
      Ij  = subproblem(j).vars_index;
      dj  = cj + rho*ATj'*max(A*x-b+lambda/rho,0);
      gc(Ij)  = dj + (0.5-x(Ij))/tau;
%       gc(Ij)  = dj;
      %         end
      %         for j = 1 : length(subproblem)
      gm.A   = subproblem(j).A;
      gm.obj = gc(subproblem(j).vars_index);
      gm.rhs = subproblem(j).rhs;
      gm.lb  = subproblem(j).lb;
      gm.ub  = subproblem(j).ub;
      gm.sense = subproblem(j).sense;
      gm.vtype = 'C';
      gm.modelsense = 'min';
      params.outputflag = 0;
      result   = gurobi(gm, params);
      x(Ij)    = result.x;
      if result.status == "INTERRUPTED"
        return
      end
    end
    if norm(x_old-x) <= eps
      break;
    end
  end


  pfeas  = norm(max(A*x-b,0));

  cx     = (model.obj)'*x;
  lobj   = cx+rho*norm(max(A*x-b+lambda/rho,0))^2/2 - lambda'*lambda/2/rho;
  error  = norm(xk-x)/norm(xk);

  fprintf("%+.2d %+.2e %+.2e, %+.3e %+.3e %+.3e  %.2d  %d\n", ...
    k, cx, lobj, pfeas, error, rho,tau, iter);
  if pfeas == 0 && error < eps
    break;
  end
  rho    = sigma*rho;
  lambda = max(0,lambda + rho*(A*x-b));
end
end