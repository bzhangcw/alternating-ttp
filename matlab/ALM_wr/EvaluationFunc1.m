function Funcollect=EvaluationFunc1

Funcollect.palm_l=@palm_l;

Funcollect.palm_dual=@palm_dual;

end


%% ALM-oringnal (dj  = cj + ATj'*(lambda+rho*max(Axm-b/2,0)))

function [x,k] = palm_dual(subproblem,coupling,model)
warning('off');
A  = coupling.A;
b  = coupling.rhs;

eps    = 1e-6;
[m,n]  = size(A);

x      = zeros(n,1);
rho    = 1e-4;
lambda = rho*zeros(m,1);

rho0   = 1e-4;
sigma  = 2;
Anorm  = normest(subproblem(1).coupling);
%Anorm  = normest(A);
tau    = 1/(Anorm^2*rho0);
kmax   = 1000;
imax   = 1;
tic
headers = ["c'x","lobj","|Ax - b|","error","rho","tau","kl","t"];
slots = ["%10s","%10s","%14s","%8s","%10s","%9s","%9s","%6s","%9s"];
header = 'k';
for j=1:8
    header=strcat(header, sprintf(slots(j), headers(j)));
end
header = strcat(header, '\n');
fprintf(header);

Ax=A*x;
for k  = 1 : kmax

    xk = x;
%    nobj = model.obj'*x+lambda'*(Ax-b)+rho*norm(max(Ax-b,0))^2/2;
    for iter = 1 : imax
        x_old = x;
        for j = 1 : length(subproblem)
            cj  = subproblem(j).c;
            Ij  = subproblem(j).vars_index;
            ATj = subproblem(j).coupling;
            Axm = Ax - ATj*x(Ij);
            dj  = cj + ATj'*(lambda+rho*max(Axm-b/2,0));
            gc  = tau* dj;
%            dj  = cj + ATj'*lambda+rho*ATj'*max(Ax-b,0);
%            gc  = tau*dj+(0.5-x(Ij))  ;
            
                        
            gm.A   = subproblem(j).A;
            gm.obj = gc;
            gm.rhs = subproblem(j).rhs;
            gm.lb  = subproblem(j).lb;
            gm.ub  = subproblem(j).ub;
            gm.sense = subproblem(j).sense;
%            gm.sense = '=';
            gm.vtype = 'C';
            gm.modelsense = 'min';
            params.outputflag = 0;
            result   = gurobi(gm, params);
            xj       = result.x;
            
            if norm(xj-x_old(Ij))>eps
                Ax       = Ax+ATj*(xj-x_old(Ij));
                x(Ij)    = xj;
            end
            
        end
        
%         nd     = norm(model.obj + A'*lambda+ rho*A'*max(Ax-b,0));
%         nobj   = model.obj'*x+lambda'*(Ax-b)+rho*norm(max(Ax-b,0))^2/2;
%         fprintf("%+.2d %.3e %.3e\n",  iter, nd, nobj_old);
        if norm(x_old-x) <= eps
            break;
        end
    end

    
    gtime=toc;
    Axb=Ax-b;
    pfeas  = norm(max(Axb,0));
%    alpha  = 0.001;
    %    alpha  = 0.0001/(sqrt(2)*k*pfeas);
    %     lambda = max(0,lambda+alpha*Axb);
    %     rho    = rho+alpha*pfeas^2/2;
    lambda = max(0,lambda + rho*(A*x-b));
    rho    = sigma*rho;
    
    cx     = (model.obj)'*x;
    lobj   = cx+lambda'*Axb+rho*pfeas^2/2;
    error  = norm(xk-x)/(norm(xk)+1e-3);
    
    fprintf("%+.2d %+.2e %+.2e, %+.3e %+.3e %+.3e %+.2e %.3d %.1e\n", ...
        k,cx,lobj,pfeas,error,rho,tau,iter,gtime);
    if pfeas == 0 
        break;
    end
%     if error < eps
%         lambda = max(0,lambda+rho*Axb);
%         rho    = rho*sigma;
%     end
end

end

%% ALM-proxlinear (stepsize tau)
function [x] = palm_l(subproblem,coupling,model)
warning('off');
A  = coupling.A;
b  = coupling.rhs;

eps    = 1e-6;
[m,n]  = size(A);
x      = ones(n,1);
rho    = 1e-4;
lambda = rho*ones(m,1);

rho0   = 1e-4;
sigma  = 2;
Anorm  = normest(subproblem(1).coupling);
%Anorm  = normest(A);
tau    = 1/(Anorm^2*rho0);
kmax   = 1000;
imax   = 10;
tic
headers = ["c'x", "lobj", "|Ax - b|", "error", "rho","tau","kl"];
slots = ["%10s", "%10s", "%14s", "%8s", "%10s", "%9s","%5s"];
header = 'k';
for j=1:7
    header=strcat(header, sprintf(slots(j), headers(j)));
end
header = strcat(header, '\n');
fprintf(header);

Ax=A*x;
for k  = 1 : kmax

    xk = x;
    nobj = model.obj'*x+lambda'*(Ax-b)+rho*norm(max(Ax-b,0))^2/2;
    for iter = 1 : imax
        x_old = x;
        nobj_old = nobj;
        for j = 1 : length(subproblem)
            cj  = subproblem(j).c;
            Ij  = subproblem(j).vars_index;
            ATj = subproblem(j).coupling;
            dj  = cj + ATj'*lambda+rho*ATj'*max(Ax-b,0);
                        
            gc  = tau*dj + (0.5-x(Ij));
            
            gm.A   = subproblem(j).A;
            gm.obj = gc;
            gm.rhs = subproblem(j).rhs;
            gm.lb  = subproblem(j).lb;
            gm.ub  = subproblem(j).ub;
            gm.sense = subproblem(j).sense;
%            gm.sense = '=';
            gm.vtype = 'C';
            gm.modelsense = 'min';
            params.outputflag = 0;
            result   = gurobi(gm, params);
            xj       = result.x;
            
            if norm(xj-x_old(Ij))>eps
                Ax       = Ax+ATj*(xj-x_old(Ij));
                x(Ij)    = xj;
            end
            
        end
        
        nd     = norm(model.obj + A'*lambda+ rho*A'*max(Ax-b,0));
        nobj   = model.obj'*x+lambda'*(Ax-b)+rho*norm(max(Ax-b,0))^2/2;
        fprintf("%+.2d %.3e %.3e\n",  iter, nd, nobj_old);
        if norm(x_old-x) <= eps || abs(nobj_old-nobj)<eps
            break;
        end
    end
    gtime=toc;
      
    Axb=Ax-b;
    pfeas  = norm(max(Axb,0));
%     alpha  = 0.001;
%     %    alpha  = 0.0001/(sqrt(2)*k*pfeas);
%     lambda = max(0,lambda+alpha*Axb);
%     rho    = rho+alpha*pfeas^2/2;
    lambda = max(0,lambda + rho*(A*x-b));
    rho    = sigma*rho;
    if k==4
        tau=tau/5;
    end
    if k==7
        tau=tau/5;
    end
%         tau=tau/1.1;

    cx     = (model.obj)'*x;
    lobj   = cx+lambda'*Axb+rho*pfeas^2/2;
    error  = norm(xk-x)/(norm(xk)+1e-3);
    
    fprintf("%+.2d %+.2e %+.2e, %+.3e %+.3e %+.3e %+.2e %d %.1e\n", ...
        k,cx,lobj,pfeas,error,rho,tau,iter,gtime);
    if pfeas == 0 
        break;
    end
%     if error < eps
%         lambda = max(0,lambda+rho*Axb);
%         rho    = rho*sigma;
%     end
end


end



