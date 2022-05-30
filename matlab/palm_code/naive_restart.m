function [gc] = naive_restart(x, Ij, gc)
    Ij_select = x(Ij) > 0;
    if sum(Ij_select) > 0  % train j is selected
        gc(Ij_select) = gc(Ij_select) + 1;
    else
        gc = gc - 1;
    end
end