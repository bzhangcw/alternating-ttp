% Model
% min c'x s.t. Ax<=b, Bx<=d, x \in {0,1}
% min c'x+rho*\|max{Ax-b+lambda/rho,0}\|^2 s.t. Bx<=d, x \in {0,1}

function [x] = palm_restart(subproblems,coupling,model)
A  = coupling.A;
b  = coupling.rhs;

eps    = 1e-6;
[m,n]  = size(coupling.A);
x      = ones(n,1);
rho    = 1e-4;
lambda = rho*ones(m,1);
errorold  = 1e6;
sigma  = 1.1;
tau    = 100;
kmax   = 100;
imax   = 50;
subproblem = values(subproblems);
restart = false;
x_best  = -ones(n, 1);

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
            cj  = subproblem{j}.c;
            Ij  = subproblem{j}.vars_index;
            ATj = A(:,Ij);
            dj  = cj + rho*ATj'*max(A*x-b+lambda/rho,0);
            gc  = dj + (0.5-x(Ij))/tau;

            if restart
                Ij_select = x_old(Ij) > 0;
                if sum(Ij_select) > 0  % train j is selected
                    gc(Ij_select) = gc(Ij_select) + 1;
                else
                    gc = gc - 1;
                end
            end

            gm.A   = subproblem{j}.A;
            gm.obj = gc;
            gm.rhs = subproblem{j}.rhs;
            gm.lb  = subproblem{j}.lb;
            gm.ub  = subproblem{j}.ub;
            gm.sense = subproblem{j}.sense;
            gm.vtype = 'B';
            gm.modelsense = 'min';
            params.outputflag = 0;
            result   = gurobi(gm, params);
            x(Ij)    = result.x;

            if restart && norm(x_old-x) > eps
                fprintf(".")  % succeed
                restart = false;
            end
        end
        if norm(x_old-x) <= eps  % x_{k+1} = x_{k}
            if iter < imax
                % restart
                restart = true;
                fprintf("R")
            else
                break;
            end
        end
    end
    lambda = max(0,lambda + rho*(A*x-b));
    pfeas  = norm(max(A*x-b,0));
    cx     = (model.obj)'*x;
    lobj   = cx+rho*norm(max(A*x-b+lambda/rho,0))^2/2;
    error  = norm(xk-x)/norm(xk);

    if pfeas > 1e-4
      rho  = sigma*rho;
    end

    if pfeas < eps && cx < model.obj'*x_best
        x_best = x;
        sig = '(*)';
    else
        sig = '';
    end    
    
    fprintf("%+.2d %+.2e %+.2e, %+.3e %+.3e %+.3e  %.2d  %d %s\n", ...
        k, cx, lobj, pfeas, error, rho,tau, iter, sig);

    if pfeas <= 1e-4 && error < eps
        break;
    end
end


end