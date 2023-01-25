import numpy as np
import pickle

from solver import linear_constraint#, objective
from water_value import WvFunction

  
class Bender_cuts(WvFunction):
    def __init__(self, next_state_variable, name = "", param = [] ):
        #self.system = system
        self.next_state_variable = next_state_variable
        self.var_name = ['water_value'] + self.next_state_variable
        self.cuts_list = []
        self.n_cuts = 0
        self.name  = name
        
    def get_constraints(self):
        return self.cuts_list

    def create_cut(self, intercept, slope):
        return linear_constraint( self.var_name, [- 1.] + slope.tolist() , 'L', - intercept )

    def add_cuts(self, cuts):
        new_cuts = [c for c in cuts if c is not None ]
        self.cuts_list.extend(new_cuts)
        self.n_cuts += len(new_cuts)

    def reset_wv(self):
        self.cuts_list = []
        self.n_cuts = 0

    def copy(self, bender_cut):
        self.cuts_list = bender_cut.cuts_list
        self.n_cuts = bender_cut.n_cuts

    def save(self, string):
        path = "wv_saved/cut_param_" + string + ".dat"
        f = open( path, "w")
        pickle.dump(self,f)
        f.close()

    def load(self, string):
        path = "wv_saved/cut_param_" + string + ".dat"
        f = open( path, "r")
        bender_cut = pickle.load(f)
        f.close()
        
        self.copy(bender_cut)                
