% Model
% min c'x s.t. Ax<=b, Bx<=d, x \in {0,1}
% min c'x+rho*\|max{Ax-b+lambda/rho,0}\|^2 s.t. Bx<=d, x \in {0,1}
% indefinite proximal version,
% includes an extrapolation step.
% 线性近似不准，是不是应该先小步子再大步子调
function [x] = ipalm(subproblems,coupling,model)
A  = coupling.A;
b  = coupling.rhs;

% est of 2-norm
sigA   = normest(coupling.A'*coupling.A);

eps    = 1e-6;
[m,n]  = size(coupling.A);
x      = ones(n,1);
rho    = 1e-3;
lambda = rho*ones(m,1);
sigma  = 1.1;
tau    = 100;
kmax   = 300;
imax   = 30;
subproblem = values(subproblems);

headers = ["c'x", "lobj", "|Ax - b|", "error", "rho", "tau", "kl"];
slots = ["%10s", "%10s", "%14s", "%8s", "%10s", "%10s", "%6s"];
header = ' k';
for j=1:7
    header=strcat(header, sprintf(slots(j), headers(j)));
end
header = strcat(header, '\n');
fprintf(header);
for k = 1 : kmax
    xk=x;
    lambda = max(0,lambda + rho*(A*x-b));
    for iter = 1 : imax
        x_old = x;
        for j = 1 : length(subproblem) %
            cj  = subproblem{j}.c;
            Ij  = subproblem{j}.vars_index;
            ATj = A(:,Ij);
            dj  = cj + rho*ATj'*max(A*x-b+lambda/rho,0);
            gc(Ij)  = dj + (0.5-x(Ij))/(2*sigA*rho);
%         end
%         for j = 1 : length(subproblem)
            gm.A   = subproblem{j}.A;
            gm.obj = gc(subproblem{j}.vars_index);
            gm.rhs = subproblem{j}.rhs;
            gm.lb  = subproblem{j}.lb;
            gm.ub  = subproblem{j}.ub;
            gm.sense = subproblem{j}.sense;
            gm.vtype = 'C';
            gm.modelsense = 'min';
            params.outputflag = 0;
            result   = gurobi(gm, params);
            if result.status == "INTERRUPTED"
              return
            end 
            x(Ij)    = result.x;
        end
        if norm(x_old-x) <= eps
            break;
        end
    end
    lambda = lambda + rho*A*(x-xk);
    
    pfeas  = norm(max(A*x-b,0));
    if pfeas > 1e-4
      rho  = sigma*rho;
    end
    cx     = (model.obj)'*x;
%     lobj   = cx+rho*norm(max(A*x-b+lambda/rho,0))^2/2;
    lobj   = cx + rho/2*pfeas^2 + lambda'*(A*x - b);
    error  = norm(xk-x)/norm(xk);

    fprintf("%+.2d %+.2e %+.2e, %+.3e %+.3e %+.3e %+.2e %d\n", ...
        k, cx, lobj, pfeas, error, rho, tau, iter);
%     if pfeas <= 1e-4 %&& error < eps
%         break;
%     end
end


end