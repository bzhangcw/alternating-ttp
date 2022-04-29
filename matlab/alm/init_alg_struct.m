
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