import numpy as np

import ctypes
import glob
import os
import time

from pyepo.model.opt import optModel

import pyomo.environ as pyo
from pyomo.environ import value
import pyomo.kernel as pmo

class GH2_d(optModel):

    def __init__(self,T,d,S,n_on,pieces,coeff,gap): 
                
        self.T=T
        self.d=d
        self.S=S
        self.pieces = pieces
        self.coeff = coeff
        self.N_tf = len(pieces)
        self.n_on = n_on
        
        self.cost = None
        
        self.obj = None
        
        self.gap = gap
        
        super().__init__()

    def _getModel(self):
        """
        A method to build model

        Returns:
            tuple: optimization model and variables
        """

        model = pyo.ConcreteModel()
    

        #Model parameters
        model.time = pyo.RangeSet(0,self.T-1)
        model.tf = pyo.RangeSet(0,self.N_tf-1)

        #Model variables
        model.x = pyo.Var(model.time, domain = pyo.NonNegativeReals)
        model.z = pyo.Var(model.time, domain = pyo.Binary)
        model.x_tf = pyo.Var(model.time* model.tf, domain = pyo.NonNegativeReals)
        model.y = pyo.Var(model.time, domain = pyo.NonNegativeReals)
        model.I = pyo.Var(model.time, domain = pyo.NonNegativeReals)
                      
        #objective
        model.o = pyo.Objective(rule= 0 )
                      
        #Constraints        
        @model.Constraint(model.time)
        def demand(m,t):
            if t>0:
                return model.I[t-1] + model.y[t]  == self.d[t] + model.I[t]
            else:
                return model.y[0] == self.d[0] + model.I[0]
        @model.Constraint(model.time)
        def capa_storage(m,t):
            return model.I[t]  <= self.S[t]
            
        @model.Constraint(model.time)
        def production_function(m,t):
            return model.y[t] == model.z[t] * self.coeff[0] + sum( model.x_tf[t, n + 1] * (self.coeff[n + 1] - self.coeff[n]) * 1.0 / (self.pieces[n + 1] - self.pieces[n]) for n in range(self.N_tf - 1)  )

        @model.Constraint(model.time)
        def piecewise_power_consumption(m,t):
            return model.x[t] == sum( model.x_tf[t, n] for n in model.tf  )

        @model.Constraint(model.time, model.tf)
        def piecewise_power_capacity(m,t,n):
            if n==0:
                return  model.x_tf[t, 0] == model.z[t] * self.pieces[0] 
            else:
                return  model.x_tf[t, n] <= model.z[t] * (self.pieces[n] - self.pieces[n-1])            

        @model.Constraint(model.time)
        def activation_cstr(m,t):
            if t>0:
                return sum( model.z[t2] for t2 in range(t,min(self.T,t+self.n_on))) >= min(self.n_on,self.T-t)*(model.z[t]-model.z[t-1])
            else:
                return pyo.Constraint.Skip
                          
        sol = np.zeros(self.T, dtype=np.int32)     
                          
        return model, sol

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (ndarray): cost of objective function
        """

        self.cost = np.array(c, dtype=np.float32)

        #Objective
        self._model.del_component(self._model.o)
        self._model.o = pyo.Objective(rule= sum( self._model.x[t]*self.cost[t] for t in self._model.time) )
        

    def solve(self):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """

        opt = pyo.SolverFactory('cplex')
        opt.options['mipgap'] = self.gap
        opt.options['timelimit'] = 10e5
        results = opt.solve(self._model,tee=False)

        obj = value(self._model.o)
        sol = np.array([ self._model.x[t]() for t in range(self.T)])
                
        return sol, obj

"""
#Define optmodel
T = 24
C= 15
d = np.array(np.ones(T)*0.3*C)
S = np.array(np.ones(T)*2.5*C)

n_on = 6
pieces = [250,750,1000]
coeff = [3,12,15]

optmodel = GH2_d(T,d,S,n_on,pieces,coeff,0.0001)

cost = np.random.rand(T)
optmodel.setObj(cost) # set objective function
sol, obj = optmodel.solve() # solve
print(obj)
print(sol)
"""