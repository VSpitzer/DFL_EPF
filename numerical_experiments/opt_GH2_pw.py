import numpy as np

import ctypes
import glob
import os
import time

from pyepo.model.opt import optModel

import pyomo.environ as pyo
from pyomo.environ import value
import pyomo.kernel as pmo

class GH2_pw(optModel):

    def __init__(self,T,N_d,tab_ta,tab_td,d,n_on,pieces,coeff,gap): 
                
        self.T=T
        self.N_d = N_d
        self.tab_ta = tab_ta
        self.tab_td = tab_td
        self.d=d
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
        model.pw = pyo.RangeSet(0,self.N_d-1)
        model.tf = pyo.RangeSet(0,self.N_tf-1)

        #Model variables
        model.x = pyo.Var(model.time, domain = pyo.NonNegativeReals)
        model.z = pyo.Var(model.time, domain = pyo.Binary)
        model.x_tf = pyo.Var(model.time* model.tf, domain = pyo.NonNegativeReals)
        model.y = pyo.Var(model.time, domain = pyo.NonNegativeReals)
        model.ypw = pyo.Var(model.time*model.pw, domain = pyo.NonNegativeReals)
        model.I = pyo.Var(model.time*model.pw, domain = pyo.NonNegativeReals)
                      
        #objective
        model.o = pyo.Objective(rule= 0 )
                      
        #Constraints        
        @model.Constraint(model.time,model.pw)
        def balance_of_plant(m,t,pw):
            if t>0:
                return model.I[t-1,pw]  + model.ypw[t,pw]  == model.I[t,pw]
            else:
                return model.ypw[0,pw]  == model.I[0,pw]
                
                
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
        def prod_2_inv(m,t):
            return model.y[t]  == sum(model.ypw[t,pw] for pw in model.pw)
        @model.Constraint(model.time,model.pw)
        def no_storage(m,t,pw):
            if t<self.tab_ta[pw]:
                return model.I[t,pw] == 0
            elif t>=self.tab_td[pw]:
                return model.I[t,pw] == self.d[pw]
            else:
                return pyo.Constraint.Skip

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
T = 3*24
C= 15
N_pw = 4
tab_ta = np.array([t*12+6 for t in range(N_pw)])
tab_td = np.array([6+2*12+t*12 for t in range(N_pw)])
d = np.round(np.array([ 0.7*12 for n in range(N_pw) ])*C,0)

n_on = 6
pieces = [250,750,1000]
coeff = [3,12,15]
cost = np.random.rand(T)

myoptmodel = GH2_pw(T,N_pw,tab_ta,tab_td,d,n_on,pieces,coeff,0.0001)
myoptmodel.setObj(cost) # set objective function
sol, obj = myoptmodel.solve() # solve
print(obj)
print(sol)
"""