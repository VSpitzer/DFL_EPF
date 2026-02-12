import numpy as np

import ctypes
import glob
import os
from ctypes import CDLL  
from _ctypes import LoadLibrary  

from pyepo.model.opt import optModel

class LotSizingModel(optModel):

    def __init__(self, T, C, x_on, d, S, s_i):
        """
        Args:
            T (int): Time horizon
            C (int): Production capacity
            x_on (int): Setup consumption
            d (ndarray(T)): Demand
            S (ndarray(T)): Storage capacity
            s_i (float): Initial storage value
        """
        self.T = T
        self.C = C
        self.x_on = x_on
        self.s_i = s_i
        self.d = np.array(d, dtype=np.int32)
        self.S = np.array(S, dtype=np.int32)
        self.cost = np.ones(T, dtype=np.float32)
        
        
        super().__init__()

    def _getModel(self):
        """
        A method to build model

        Returns:
            tuple: optimization model and variables
        """
        #Precompile C++ poly file
        handle = LoadLibrary("C:\\Users\\Victor\\Documents\\Code\\DFL_EPF\\DFL_toolbox\\lib_LS_CC_SUB_T3.so")
        lib = CDLL(name="C:\\Users\\Victor\\Documents\\Code\\DFL_EPF\\DFL_toolbox\\lib_LS_CC_SUB_T3.so",handle = handle)

        lib.CLSP_T3.restype = ctypes.c_void_p
        lib.CLSP_T3.argtypes = [np.ctypeslib.ndpointer(dtype=np.int32),
                                       ctypes.c_int,
                                       np.ctypeslib.ndpointer(dtype=np.float32),
                                       ctypes.c_int,
                                       np.ctypeslib.ndpointer(dtype=np.int32),
                                       np.ctypeslib.ndpointer(dtype=np.int32),
                                       ctypes.c_int,
                                       ctypes.c_int
                                       ]
       
        sol = np.zeros(self.T, dtype=np.int32)     

        model = {"lib":lib, "sol":sol}
                                       
        return model, sol

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (ndarray): cost of objective function
        """
        self.cost = np.array(c, dtype=np.float32)

    def solve(self):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """

        #solution
        self._model["lib"].CLSP_T3(self._model["sol"], self.T, self.cost, self.s_i, self.S, self.d, self.C, self.x_on)
        
        #Optimal sol and objective
        sol = np.round(np.array(self._model["sol"]),3)
        obj = sum( sol[t]*self.cost[t] for t in range(self.T))
        return sol, obj

