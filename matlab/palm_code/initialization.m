
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
%            coupling_constrs_index = int64(data{key}) + 1;
            coupling_constrs_index = cellfun(@int64,cell(data{key}))+1;
            
            % create coupling struct
            coupling.A = model.A(coupling_constrs_index , :);
            coupling.rhs = model.rhs(coupling_constrs_index , :);
            coupling.constrs_index = coupling_constrs_index;
            continue
        end
            % get var and constr for each subproblem(train)
%            vars_index = int64(data{key}{"vars"}) + 1;
            vars_index = cellfun(@int64,cell(data{key}{"vars"}))+1;
%            constrs_index = int64(data{key}{"constrs"}) + 1;
            constrs_index = cellfun(@int64,cell(data{key}{"constrs"})) + 1;
            
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
