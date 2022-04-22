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
function [alg, info] = linearized_alm(model, filename, params)
    [subproblems, coupling] = initialization(filename, model);
    
    % create algorithm struct
    alg = init_alg_struct(model, coupling, params);

    headers = ["c'x", "phi", "|Dx - d|", "gap", "rho"];
    slots = ["%10s", "%10s", "%13s", "%8s", "%10s"];
    header = 'k';
    for j=1:5
      header=strcat(header, sprintf(slots(j), headers(j)));
    end
    header = strcat(header, '\n');

    % outer loop of alm
    for iter = 2:alg.iter_max+1  % iter starts from 2 to itermax+1
        if ~mod(iter - 2, 50)
            fprintf(header);
        end
        % reset phik
        phik = 0;

        alg.iter = iter;  % update iter
        % copy prev iter value to cur iter
        alg.x(:, alg.iter) = alg.x(:, alg.iter-1);
        alg.d(:, alg.iter) = alg.d(:, alg.iter-1);
        
        for x_iter = 1:alg.x_outer_iter_max
            % update x for all trains
            for subproblem = values(subproblems)
                subproblem = subproblem{1};
                [alg, r, ret_code] = update_and_save_x(subproblem, coupling, alg);
                
                if ret_code == 0
                    phik = phik + r.objval;
                end
            end
            
            % break if x_k is not changing
            if norm(alg.x(:, alg.iter) - alg.x(:, alg.iter-1)) < 1e0
                break
            end
        end
        alg = update_and_save_lambda(coupling, alg);  % update lagrange multipliers
        alg = update_and_save_d(coupling, alg);  % update gradient of augmented term
        
        % print status
        xk = alg.x(:, alg.iter);
        zk = model.obj' * xk;
        gapk = 100*abs(zk-phik)/(abs(phik)+1e-6);
        psub = max(coupling.A*xk  - coupling.rhs, 0);
        pfeas = norm(psub);
        fprintf("%+.2d %+.2e %+.2e %+.3e %+.1e%% %+.3e\n", ...
        iter-1, zk, phik, pfeas, gapk, alg.rho);
        info = struct;  % init empty struct
        if pfeas < 1e-6 && gapk < 1e-6
            % collect information
            info.z = zk;
            info.phik = phik;
            info.gap = gapk;
            info.pfeas = norm(psub);
            break
        end

        % update alg struct end of each iter
        alg = update_alg(alg);
    end
end

function [r] = update_x(subproblem, xk_j, dk_j, rho, tau, params) 
    % calculate new obj
    subproblem.obj = subproblem.c + rho * dk_j + 0.5 / tau * (ones(size(xk_j)) - 2*xk_j);
    subproblem.objcon = - rho * dk_j' * xk_j + 0.5 / tau * (xk_j'*xk_j);

    % solve with gurobi
    r = gurobi(subproblem, params);
end

function [alg, r, ret_code] = update_and_save_x(subproblem, coupling, alg)
    % get d and xk
    vars_index = subproblem.vars_index;
    
    % get prev iter value
    xk_j = alg.x(vars_index, alg.iter);
    dk_j = alg.d(vars_index, alg.iter);
    
    % update x for single train j several times
    for iter = 1:alg.x_inner_iter_max
        % get new xk
        r = update_x(subproblem, xk_j, dk_j, alg.rho, alg.tau, alg.params);
        
        % get opt result
        if strcmp(r.status, "OPTIMAL")
            xk_j = r.x;
            alg.x(vars_index, alg.iter) = xk_j;  % update xk_j
            dk = update_d(coupling, alg.x(:, alg.iter), alg.lambda(:, alg.iter-1), alg.rho);
            dk_j = dk(vars_index, 1);
            alg.d(:, alg.iter) = dk;  % update dk
            ret_code = 0;
        else
            disp("Error in optimization!")
            ret_code = -1;
            break
        end
    end
end

function [lk_new] = update_lambda(coupling, alg)
    xk = alg.x(:, alg.iter);  % x_{k+1}
    A = coupling.A;
    b = coupling.rhs;
    lk = alg.lambda(:, alg.iter-1);

    lk_new = max(lk + alg.rho * (A * xk - b), 0);
end

function [alg] = update_and_save_lambda(coupling, alg)
    alg.lambda(:, alg.iter) = update_lambda(coupling, alg);
end

function [dk] = update_d(coupling, xk, lk, rho)
    A = coupling.A;
    b = coupling.rhs;
    dk = A'*max(A*xk-b+lk/rho, 0); % update d_k
end

function [alg] = update_and_save_d(coupling, alg)
    xk = alg.x(:, alg.iter);
    lk = alg.lambda(:, alg.iter);
    alg.d(:, alg.iter) = update_d(coupling, xk, lk, alg.rho);
end

function [alg] = update_alg(alg)
    alg.rho = alg.rho / alg.iter;
end

function [subproblems, coupling] = initialization(filename, model)
    % pe = pyenv("Version", "E:\APPLICATION\Anaconda3\envs\rail\python.exe");
    fid = py.open(filename,'rb');
    data = py.pickle.load(fid);
    
    % containers.Map store all subproblems
    subproblems = containers.Map('KeyType','int32', 'ValueType','any');

    for raw_key = py.list(keys(data))
        key = raw_key{1};
        try 
            % convert to double
            key = double(key);
        catch ME
            % get coupling constraints
            coupling_constrs_index = int64(data{key}) + 1;
            
            % create coupling struct
            coupling.A = model.A(coupling_constrs_index , :);
            coupling.rhs = model.rhs(coupling_constrs_index , :);
            coupling.constrs_index = coupling_constrs_index;
            continue
        end
            % get var and constr for each subproblem(train)
            vars_index = int64(data{key}{"vars"}) + 1;
            constrs_index = int64(data{key}{"constrs"}) + 1;
            
            % create subproblem struct
            subproblem.A = model.A(constrs_index, vars_index);
            
            % sanity check
            %remain_constrs_index = setdiff(1:size(model.A,1), constrs_index);
            %assert(all(model.A(remain_constrs_index, vars_index) == 0, 'all'))
            remain_vars_index = setdiff(1:size(model.A,2), vars_index);
            assert(sum(model.A(constrs_index, remain_vars_index), 'all') == 0)
            % no obj
            subproblem.sense = model.sense(constrs_index);
            subproblem.rhs = model.rhs(constrs_index);
            subproblem.lb = model.lb(vars_index);
            subproblem.ub = model.ub(vars_index);
            subproblem.vtype = model.vtype(vars_index);
            try
                subproblem.modelsense = model.modelsense;
            catch
                subproblem.modelsense = 'min';
            end
            subproblem.varnames = model.varnames(vars_index);
            subproblem.constrnames = model.constrnames(constrs_index);
    
            % save aux attr
            subproblem.c = model.obj(vars_index);
            subproblem.vars_index = vars_index;
            subproblem.constrs_index = constrs_index;
            subproblem.No = key;
    
            % save subproblem
            subproblems(key) = subproblem;
    end    
end

function [alg] = init_alg_struct(model, coupling, params)
    [m, n] = size(model.A);
    rho = 1e+1;
    tau = 1e+1;

    % define algorithm struct
    alg = struct;
    alg.iter_max = 100;  
    alg.iter = 1; % current iteration index
    alg.x_outer_iter_max = 1;  % update all js x_j multiple times before update multiplier
    alg.x_inner_iter_max = 1;  % update single j x_j multiple times 
    alg.m = m;
    alg.n = n;
    alg.lambda = -ones(length(coupling.constrs_index), alg.iter_max+1);
    alg.d = -ones(n, alg.iter_max+1);  % index=1 is init value
    alg.x = -ones(n, alg.iter_max+1);  % index=1 is init value
    alg.rho = rho;
    alg.tau = tau;
    alg.params = params;

    % init
    alg.lambda(:, 1) = rand(size(alg.lambda, 1), 1);
    alg.d(:, 1) = rand(size(alg.d, 1), 1);  % index=1 is init value
    alg.x(:, 1) = rand(size(alg.x, 1), 1);  % index=1 is init value
end