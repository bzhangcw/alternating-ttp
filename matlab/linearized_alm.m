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
        alg.z(:, alg.iter) = alg.z(:, alg.iter-1);
        alg.d(:, alg.iter) = alg.d(:, alg.iter-1);
        alg.lambda(:, alg.iter) = alg.lambda(:, alg.iter-1);

        for x_iter = 1:alg.x_outer_iter_max
            % update d (gradient of augmented term)
            alg = update_and_save_d(coupling, alg);

            % update x for all trains
            for subproblem = values(subproblems)
                subproblem = subproblem{1};
                [alg, r, ret_code] = update_and_save_x(subproblem, coupling, alg);
                
%                 if ret_code == 0
%                     phik = phik + r.objval;
%                 end
            end
            
            % break if x_k is not changing
%             norm_x_diff = norm(alg.x(:, alg.iter) - alg.x(:, alg.iter-1));
%             if norm_x_diff < 1e-3
%                 break
%             end
            if alg.debug
                alm_obj = cal_alm(model, coupling, alg);
                fprintf("alm obj: %.2e\n", alm_obj)
            end
        end
        if alg.debug
            norm_x_diff = norm(alg.x(:, alg.iter) - alg.x(:, alg.iter-1));
            fprintf("norm(x_{k+1} - x_k)=%.2e\n", norm_x_diff)
        end

        % update slack variables
        alg = update_and_save_z(coupling, alg);
        % update lagrange multipliers
        alg = update_and_save_lambda(coupling, alg);
        
        % print status
        xk = alg.x(:, alg.iter);
        zk = model.obj' * xk;
        phik = cal_alm(model, coupling, alg);
        gapk = 100*abs(zk-phik)/(abs(phik)+1e-6);
        psub = max(coupling.A*xk  - coupling.rhs, 0);
        pfeas = norm(psub);
        fprintf("%+.2d %+.2e %+.2e %+.3e %+.1e%% %+.3e\n", ...
        iter-1, zk, phik, pfeas, gapk, alg.rho);
        
        pfeasb = alg.pfeas;  % get pfeas before
        alg.pfeas = pfeas;  % update pfeas
        if pfeas > pfeasb
            alg.rho = 1.2 * alg.rho;
        end

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
%             dk = update_d(coupling, alg.x(:, alg.iter), alg.lambda(:, alg.iter-1), alg.rho);
%             dk_j = dk(vars_index, 1);
%             alg.d(:, alg.iter) = dk;  % update dk
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

    if alg.debug
        fprintf("norm(l_{k+1} - l_k)=%.2e\n", norm(alg.lambda(:, alg.iter)- alg.lambda(:, alg.iter-1)))
    end
end

function [dk] = update_d(coupling, xk, lk, zk, rho)
    A = coupling.A;
    b = coupling.rhs;
    dk = A' * (A*xk-b+lk/rho+zk); % update d_k
end

function [alg] = update_and_save_d(coupling, alg)
    xk = alg.x(:, alg.iter);
    lk = alg.lambda(:, alg.iter);
    zk = alg.z(:, alg.iter);
    alg.d(:, alg.iter) = update_d(coupling, xk, lk, zk, alg.rho);

    if alg.debug
        fprintf("norm(d_{k+1} - d_k)=%.2e\n", norm(alg.d(:, alg.iter)- alg.d(:, alg.iter-1)))
    end
end

function [zk] = update_z(coupling, xk, lk, rho)
    A = coupling.A;
    b = coupling.rhs;
    zk = max(-(A*xk-b+lk/rho), 0);
end

function [alg] = update_and_save_z(coupling, alg)
    xk = alg.x(:, alg.iter);
    lk = alg.lambda(:, alg.iter);
    alg.z(:, alg.iter) = update_z(coupling, xk, lk, alg.rho);
end

function [alg] = update_alg(alg)
    alg.rho = alg.rho * alg.rho_coeff;
    alg.tau = alg.tau * alg.tau_coeff;
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

    % sanity check
    A_decoup = model.A(setdiff(1:size(model.A,1), coupling.constrs_index), :);
    for subproblem = values(subproblems)
        subproblem = subproblem{1};
        constrs_index = subproblem.constrs_index;
        vars_index = subproblem.vars_index;
        remain_constrs_index = setdiff(1:size(A_decoup,1), constrs_index);
        assert(sum(A_decoup(remain_constrs_index, vars_index), 'all') == 0)
        remain_vars_index = setdiff(1:size(A_decoup,2), vars_index);
        assert(sum(A_decoup(constrs_index, remain_vars_index), 'all') == 0)
    end
end

function [alg] = init_alg_struct(model, coupling, params)
    [m, n] = size(model.A);
    rho = 0.5;
    rho_coeff = 0.99;
    tau = 1e+5;
    tau_coeff = 0.9;

    % define algorithm struct
    alg = struct;
    alg.iter_max = 100;
    alg.iter = 1; % current iteration index
    alg.x_outer_iter_max = 10;  % update all js x_j multiple times before update multiplier
    alg.x_inner_iter_max = 1;  % update single j x_j multiple times 
    alg.m = m;
    alg.n = n;
    alg.lambda = -ones(length(coupling.constrs_index), alg.iter_max+1);
    alg.d = -ones(n, alg.iter_max+1);  % index=1 is init valuea
    alg.x = -ones(n, alg.iter_max+1);  % index=1 is init value
    alg.z = -ones(size(coupling.A, 1), alg.iter_max+1);
    alg.rho = rho;
    alg.rho_coeff = rho_coeff;
    alg.tau = tau;
    alg.tau_coeff = tau_coeff;
    alg.params = params;

    % statistics
    alg.pfeasb = 0;
    alg.pfeas = 0;

    % running mode
    alg.debug = false;

    % how to update d ?
    alg.d_update_mode = 1;  
    % 0 for d = Ax^k-b+\lambda^k/\rho + z^{k+1} = 
    % 1 for d = Ax^k-b+\lambda^k/\rho + z^{k}

    % init
%     alg.lambda(:, 1) = rand(size(alg.lambda, 1), 1) * alg.rho;
%     alg.x(:, 1) = randi([0, 1], size(alg.x, 1), 1);
    alg.z(:, 1) = zeros(size(alg.z, 1), 1);
    alg.lambda(:, 1) = 100 * ones(size(alg.lambda, 1), 1);
    alg.x(:, 1) = zeros(size(alg.x, 1), 1);
end

function [obj] = cal_alm(model, coupling, alg)
    [c, A, b, x, z, lam, rho] = deal(model.obj, coupling.A, coupling.rhs, alg.x(:, alg.iter), alg.z(:, alg.iter), alg.lambda(:, alg.iter), alg.rho);
    s = A*x-b + lam/rho + z;
    obj = c'*x + rho/2*norm(s)^2;
end