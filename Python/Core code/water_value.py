from solver import linear_constraint, objective, Linear_problem
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

 
class WvFunction(object):
    name = 'null_wv'
    
    def __init__(self):
        pass

    def get_constraints(self):
        return [linear_constraint( ['water_value'] , [1.] , 'G' , 0. )]

    def get_variable_bounds(self):
        return {'water_value' : 'free' }

    def evaluate(self, x):
        return np.zeros(len(x))

    def load(self, str):
        pass

    def save(self, str):
        pass

    def get_log(self):
        return {}

    def close(self):
        pass
    
    def plot_prediction_comparison(self, x_eval, fig = [], color_plot = 'blue'):
        if fig == [] :
            fig = plt.figure()      
            ax = fig.add_subplot(111, projection = '3d')
        else:
            ax = fig #.add_subplot(111, projection = '3d')
        
        ngrid = int(sqrt(len(x_eval)))

        xx_eval = x_eval[:,0].reshape((ngrid,ngrid))
        xy_eval = x_eval[:,1].reshape((ngrid,ngrid))

        data_Y_constraint = []
        for x in x_eval:
            data_Y_constraint.append( self.evaluate_constraint(x) )
        ax.plot_surface(xx_eval, xy_eval, np.array(data_Y_constraint).reshape((ngrid,ngrid)), color = color_plot )
        return ax

    def make_x_dataset(self, x_val = []):
        if x_val == []:
            x_val = self.data_raw_X
        x_min = np.min(x_val ,axis = 0) - 100 
        x_max = np.max(x_val ,axis = 0) + 100
        ngrid = 10
        c_ngrid = complex(0,ngrid)
        return np.mgrid[x_min[0]:x_max[0]:c_ngrid, x_min[1]:x_max[1]:c_ngrid].reshape(2,-1).T

    def evaluate_constraint(self, x):
        cons = []
        cons[:] = self.get_constraints()        
        obj = [objective( [ 1.] , ['water_value'] )]
        bounds = self.get_variable_bounds()
        bounds.update( {'water_value' : 'free'})
        [bounds.update( {var: 'free'}) for var in self.next_state_variable]
        for i, x_ in enumerate(x):
            cons.append( linear_constraint( [self.next_state_variable[i] ], [1.], 'E', x_ ) )
        linear_prob = Linear_problem('Lp_solve')
        linear_prob.populate_problem( obj, cons, bounds )
        linear_prob.solve()
        linear_prob.prob.print_problem()
        return linear_prob.get_decisions(['water_value'])[0]

    
