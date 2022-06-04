# -*- coding: utf-8 -*-
# @author: PuShanwen
# @email: 2019212802@live.sufe.edu.cn
# @date: 2021/11/24
from typing import List, Tuple, Union
from gurobipy import Model, Constr, Var


def getObjectByString(model: Model, substring: Union[str, List[str], Tuple[str, ...]],
                      var_or_constr: str, xxfix: str) -> Union[List[Constr], List[Var]]:
    """
    get grb constr.ConstrName/var.VarName start with prefix/suffix/substring
    Args:
        model: gurobi model
        substring: substring or list of substring or tuple of substring
        var_or_constr: get var or constr info
        xxfix: get prefix or suffix or substring

    Returns: list of constrs/vars match the requirements
    """
    if isinstance(substring, list):
        substring = tuple(substring)
    else:
        assert isinstance(substring, str) or isinstance(substring, tuple)
    assert var_or_constr in ("var", "constr") and xxfix in ("prefix", "substring", "suffix")

    if var_or_constr == "constr":
        if xxfix == "prefix":
            return [constr for constr in model.getConstrs() if constr.ConstrName.startswith(substring)]
        elif xxfix == "substring":
            return [constr for constr in model.getConstrs() if any(subs in constr.ConstrName for subs in substring)]
        elif xxfix == "suffix":
            return [constr for constr in model.getConstrs() if constr.ConstrName.endswith(substring)]
    elif var_or_constr == "var":
        if xxfix == "prefix":
            return [var for var in model.getVars() if var.VarName.startswith(substring)]
        elif xxfix == "substring":
            return [var for var in model.getVars() if any(subs in var.VarName for subs in substring)]
        elif xxfix == "suffix":
            return [var for var in model.getVars() if var.VarName.endswith(substring)]


def getVarByPrefix(model: Model, prefix: Union[str, List[str], Tuple[str, ...]]) -> List[Constr]:
    """
    get grb var.VarName start with prefix
    Args:
        model: gurobi model
        prefix: prefix or list of prefix or tuple of prefix

    Returns: list of vars match the requirements

    """
    return getObjectByString(model, prefix, "var", "prefix")


def getConstrByPrefix(model: Model, prefix: Union[str, List[str], Tuple[str, ...]]) -> List[Constr]:
    """
    get grb constr.ConstrName start with prefix
    Args:
        model: gurobi model
        prefix: prefix or list of prefix or tuple of prefix

    Returns: list of constrs match the requirements

    """
    return getObjectByString(model, prefix, "constr", "prefix")


def getVarBySuffix(model: Model, suffix: Union[str, List[str], Tuple[str, ...]]) -> List[Constr]:
    """
    get grb constr.ConstrName ends with suffix
    Args:
        model: gurobi model
        suffix: suffix or list of suffix or tuple of suffix

    Returns: list of constrs match the requirements

    """
    return getObjectByString(model, suffix, "var", "suffix")


def getConstrBySuffix(model: Model, suffix: Union[str, List[str], Tuple[str, ...]]) -> List[Constr]:
    """
    get grb constr.ConstrName ends with suffix
    Args:
        model: gurobi model
        suffix: suffix or list of suffix or tuple of suffix

    Returns: list of constrs match the requirements

    """
    return getObjectByString(model, suffix, "constr", "suffix")


def getVarBySubstring(model: Model, substring: Union[str, List[str], Tuple[str, ...]]) -> List[Constr]:
    """
    get grb constr.ConstrName ends with substring
    Args:
        model: gurobi model
        substring: substring or list of substrings or tuple of substrings

    Returns: list of constrs match the requirements

    """
    return getObjectByString(model, substring, "var", "substring")


def getConstrBySubstring(model: Model, substring: Union[str, List[str], Tuple[str, ...]]) -> List[Constr]:
    """
    get grb constr.ConstrName ends with substring
    Args:
        model: gurobi model
        substring: substring or list of substrings or tuple of substrings

    Returns: list of constrs match the requirements

    """
    return getObjectByString(model, substring, "constr", "substring")
