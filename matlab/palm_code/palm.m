% Model
% min c'x s.t. Ax<=b, Bx<=d, x \in {0,1}
% min c'x+rho*\|max{Ax-b+lambda/rho,0}\|^2 s.t. Bx<=d, x \in {0,1}

function [x] = palm(subproblem,coupling,model)
A  = coupling.A;
b  = coupling.rhs;

eps    = 1e-6;
[m,n]  = size(A);
x      = ones(n,1);
rho    = 1e-4;
lambda = rho*ones(m,1);

sigma  = 2;
tau    = 100;
kmax   = 1000;
imax   = 10;
% subproblem = values(subproblems);

headers = ["c'x", "lobj", "|Ax - b|", "error", "rho","tau","iter"];
slots = ["%10s", "%10s", "%14s", "%8s", "%10s", "%9s","%5s"];
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
        for j = 1 : length(subproblem)
            cj  = subproblem(j).c;
            Ij  = subproblem(j).vars_index;
            ATj = A(:,Ij);
            dj  = cj + rho*ATj'*max(A*x-b+lambda/rho,0);
            gc(Ij)  = dj + (0.5-x(Ij))/tau;
%         end
%         for j = 1 : length(subproblem)
            gm.A   = subproblem(j).A;
            gm.obj = gc(subproblem(j).vars_index);
            gm.rhs = subproblem(j).rhs;
            gm.lb  = subproblem(j).lb;
            gm.ub  = subproblem(j).ub;
            gm.sense = subproblem(j).sense;
            gm.vtype = 'B';
            gm.modelsense = 'min';
            params.outputflag = 0;
            result   = gurobi(gm, params);
            x(Ij)    = result.x;
        end
        if norm(x_old-x) <= eps
            break;
        end
    end
    lambda = max(0,lambda + rho*(A*x-b));
    rho    = sigma*rho;
    pfeas  = norm(max(A*x-b,0));
    
    cx     = (model.obj)'*x;
    lobj   = cx+rho*norm(max(A*x-b+lambda/rho,0))^2/2;
    error  = norm(xk-x)/norm(xk);
    
    fprintf("%+.2d %+.2e %+.2e, %+.3e %+.3e %+.3e  %.2d  %d\n", ...
        k, cx, lobj, pfeas, error, rho,tau, iter);
    if pfeas == 0 && error < eps
        break;
    end
end


end