"""
Function that implements the symmetric mean absolute percentage error (sMAPE) metric.
"""

# Author: Jesus Lago

# License: AGPL-3.0 License


import numpy as np
from epftoolbox.evaluation._ancillary_functions import _process_inputs_for_metrics

import pyepo
from pyepo.func.abcmodule import optModule
from pyepo.func.utlis import _solve_or_cache

import torch

def Regret(p_real, p_pred, optmodel):


    # Checking if inputs are compatible
    p_real, p_pred = _process_inputs_for_metrics(p_real, p_pred)


    # Prepare regret retrieval
    optmodule = optModule(optmodel, processes=1, solve_ratio=1, reduction="mean",dataset=None)

    # Retrieve opt decision
    wr, obj = _solve_or_cache(torch.tensor(p_real), optmodule)
    wp, obj = _solve_or_cache(torch.tensor(p_pred), optmodule)

    return (((wp - wr) * p_real).sum(axis=1)*100./(wr*p_real).sum(axis=1)).numpy()
