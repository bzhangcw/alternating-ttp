% create a compact timetabling model
%   from an LP file.
% @author: Chuwen Z.
% @date: 22/04/14
% denote the problem as,
%  max c'z
%  s.t. 
%    B * [z;x] <=/== b;
%    D * [z;x] <=/== d;
%    z in R, x in {0, 1}
% @remark:
%  x is a binary, if x=1 means the edge is used.
%   also see model notes for detail.
%  D are the binding constraints,
%   indicated by the constraint name 'multiway_*',
%  B is simply formed from a set of shortest path problem,

%%
function [m, model] = read_model (path)
  m = gurobi_read(path);
  
  for l = 1: size(m.constrnames)
    if startsWith(m.constrnames{l}, "multi")
    break
    end
  end
  l = l -1;
  model.B = m.A(1:l, :);
  model.b = m.rhs(1:l, :);
  model.D = m.A(l+1:end, :);
  model.d = m.rhs(l+1:end, :);
  model.c = m.obj;
  model.lb = m.lb;
  model.ub = m.ub;
end

