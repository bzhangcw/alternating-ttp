# -*- coding: utf-8 -*-
# @author: PuShanwen
# @email: 2019212802@live.sufe.edu.cn
# @date: 2021/11/24
import logging
from typing import List, Tuple, Union

try:
    from gurobipy import Model, Constr


    def constr_verification(model: Model, constr_name_list: Union[List[str], str], output_not_in_input: bool = False) -> \
            Tuple[List[str], List[str]]:
        """
        check the constrs in model match with the constr_prefix_list
        Args:
            model: Gurobi Model
            constr_name_list: List of constr name, convention: name[s,t] -> name
            output_not_in_input: whether to output prefixNotInInput

        Returns: prefixNotInModel: List[str], prefixNotInput: List[str]
        """
        if isinstance(constr_name_list, str):
            constr_name_list = [constr_name_list]
        elif not isinstance(constr_name_list, List):
            raise TypeError(f"constr_name_list has wrong type!")

        constrsPrefixInputSet = set([constrName.split("[")[0] for constrName in constr_name_list])

        constrsPrefixInModelSet = set([constr.ConstrName.split('[')[0] for constr in model.getConstrs()])

        prefixNotInModel = list(constrsPrefixInputSet.difference(constrsPrefixInModelSet))
        prefixNotInInput = list(constrsPrefixInModelSet.difference(constrsPrefixInputSet))

        print("These constrs not in model: ", prefixNotInModel)
        if output_not_in_input:
            print("These constrs not in input: ", prefixNotInInput)
            return prefixNotInModel, prefixNotInInput
        else:
            return prefixNotInModel


    def getConstrByPrefix(model: Model, prefix: Union[str, List[str]]) -> List[Constr]:
        """
        get grb constr.ConstrName start with prefix
        Args:
            model: gurobi model
            prefix: prefix or list of prefix

        Returns: list of constrs match the requirements

        """
        if isinstance(prefix, str):
            prefix = [prefix]
        elif not isinstance(prefix, list):
            raise TypeError(f"prefix has wrong type!")
        return [constr for constr in model.getConstrs() if strStartswith(constr.ConstrName, prefix)]


    def getConstrBySuffix(model: Model, suffix: Union[str, List[str]]) -> List[Constr]:
        """
        get grb constr.ConstrName ends with suffix
        Args:
            model: gurobi model
            suffix: suffix or list of suffix

        Returns: list of constrs match the requirements

        """
        if isinstance(suffix, str):
            suffix = [suffix]
        elif not isinstance(suffix, list):
            raise TypeError(f"suffix has wrong type!")
        return [constr for constr in model.getConstrs() if constr.ConstrName.endswith(suffix)]


    def getConstrBySubstring(model: Model, substring: Union[str, List[str]]) -> List[Constr]:
        """
        get grb constr.ConstrName ends with substring
        Args:
            model: gurobi model
            substring: substring or list of substring

        Returns: list of constrs match the requirements

        """
        if isinstance(substring, str):
            substring = [substring]
        elif not isinstance(substring, list):
            raise TypeError(f"substring has wrong type!")
        return [constr for constr in model.getConstrs() if substring in constr.ConstrName]


    def strStartswith(name: str, prefix_list: Union[str, List[str]]) -> bool:
        for prefix in prefix_list:
            if name.startswith(prefix):
                return True
        return False


    def key_order(name: str):
        if name.startswith("SECTION_NON_COND_TRANS_LIMIT_") and "TIGHT" not in name:  # put out of limit section first
            return 0
        else:
            return 1
except Exception as e:
    logging.warning("no gurobi installed")
