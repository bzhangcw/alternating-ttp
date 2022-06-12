% Model
% min c'x s.t. Ax<=b, Bx<=d, x \in {0,1}
% min c'x+lambda'*(A*x-b)+rho*\|max{Ax-b,0}\|^2 s.t. Bx<=d, x \in {0,1}

function [x] = palm_l_restart(subproblems,coupling,model)
A  = coupling.A;
b  = coupling.rhs;

eps     = 1e-6;
[m,n]   = size(A);
x       = ones(n,1);
rho     = 1e-4;
lambda  = rho*ones(m,1);
x_best  = -ones(n, 1);
alpha  = 0.001;
restart = false;

tau    = 1;
kmax   = 150;
imax   = 50;
restart_itermax = 10;
restart_mode = 1; % 0 for naive, 1 for random

subproblem = values(subproblems);

headers = ["c'x", "lobj", "|Ax - b|", "error", "rho","tau","iter"];
slots = ["%10s", "%10s", "%14s", "%8s", "%10s", "%9s","%5s"];
header = 'k';
for j=1:7
    header=strcat(header, sprintf(slots(j), headers(j)));
end
header = strcat(header, '\n');
fprintf(header);
for k  = 1 : kmax
    xk = x;
    for iter = 1 : imax
        x_old = x;
        for j = 1 : length(subproblem)
            cj  = subproblem{j}.c;
            Ij  = subproblem{j}.vars_index;

            ATj = A(:,Ij);
            dj  = cj + ATj'*lambda+rho*ATj'*max(A*x-b,0);
            gc  = dj + (0.5-x(Ij))/tau;

            if restart
                if restart_mode == 0  # naive restart
                    gc = naive_restart(x, Ij, gc);
                elseif restart_mode == 1  # random restart
                    p_lobj = @(x)(alm_obj(A, b, model.obj, x, lambda, alpha, rho));
                    gc = perturbed_restart(subproblem, j, x, restart_itermax, p_lobj);
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
    fprintf("\n")

    Axb=A*x-b;
    pfeas  = norm(max(Axb,0));
%    alpha  = 0.0001/(sqrt(2)*k*pfeas);
    lambda = max(0,lambda+alpha*Axb);
    rho    = rho+alpha*pfeas^2/2;
    
    cx     = (model.obj)'*x;
    lobj   = cx+lambda'*Axb+rho*pfeas^2/2;
    error  = norm(xk-x)/norm(xk);
    
    if pfeas < eps && cx < model.obj'*x_best
        x_best = x;
        sig = '(*)';
    else
        sig = '';
    end    
    
    fprintf("%+.2d %+.2e %+.2e, %+.3e %+.3e %+.3e  %.2d  %d %s\n", ...
        k, cx, lobj, pfeas, error, rho,tau, iter, sig);

    if pfeas == 0 && error < eps
        break;
    end
end

function [lobj] = alm_obj(A, b, c, x, lambda, alpha, rho)
    Axb=A*x-b;
    pfeas  = norm(max(Axb,0));
    lambda = max(0,lambda+alpha*Axb);
    rho    = rho+alpha*pfeas^2/2;

    cx     = c'*x;
    lobj   = cx+lambda'*Axb+rho*pfeas^2/2;
end
end