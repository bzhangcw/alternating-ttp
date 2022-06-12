function [gc_best] = random_path_restart(subproblem, j, x, itermax, alm_obj)
    Ij = subproblem{j}.vars_index;
    x_best = x;
    gc_best = subproblem{j}.c;
    obj_best = alm_obj(x_best);
    for i = 1:itermax
        % random generate path with perturbated obj
        gc = 2 * rand(size(x(Ij))) - 1;
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
        x(Ij)    = result.x;  % change x each iteration

        obj_now = alm_obj(x);
        if obj_now < obj_best
            obj_best = obj_now;
            x_best = x;
            gc_best = gc;
        end
    end
end