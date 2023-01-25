import numpy as np
import mpi4py.MPI as MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

from algo import Algo
from convex_neural_net import Convex_Relu_nn

class CFDP(Algo):
    """docstring for sddp"""
    
    algo_name = 'CFDP'
    
    def __init__(self, model, stateVariable, solver, sample_mode):
        super(CFDP, self).__init__(model,stateVariable, solver, sample_mode)        

        self.water_value_type = Convex_Relu_nn
        self.water_value_function = self.create_water_value_function()
      

    def initialize(self, algo):
        self.n_forward_pass = algo.n_forward_pass 
        self.benefit_record[:self.n_forward_pass+1] = algo.benefit_record[:algo.n_forward_pass+1]
        self.state_variable_record[:self.n_forward_pass+1] = algo.state_variable_record[:algo.n_forward_pass+1]
        self.dual_variable_record[:self.n_forward_pass+1] = algo.dual_variable_record[:algo.n_forward_pass+1]

        for t in range(self.model.params.T-1, -1, -1):
            self.algo_ts = t
            for i_train in range(300):
                self.fit_cvx_nn(t)
                self.water_value_function[t].iteration = 0
            self.water_value_function[t].update_standardized_value = False


    def backward_pass(self):
        if self.n_forward_pass % 10 == 0 :
            for t in range(self.model.params.T-1, -1, -1):
                self.algo_ts = t
                
                for i_train in range(1):
                    self.fit_cvx_nn(t)
                
        return 1 


    def get_wv_variable_bounds(self):
        var_bounds = super(CFDP,self).get_wv_variable_bounds()
        var_bounds.update( self.water_value_function[self.algo_ts+1].get_variable_bounds() )
        return var_bounds

    def fit_cvx_nn(self,t):
        self.water_value_function[t].data_raw_X, self.water_value_function[t].data_raw_Dx, self.water_value_function[t].data_raw_Y = self.get_training_set(t)
        self.water_value_function[t].train()

    def get_training_set_new(self, t):
        x = []
        y = []
        for data in self.water_value_function[self.algo_ts].data_training :
            x.append(data['x'])
            y.append(data['ts_benefit'] + self.discount_factor * 
                self.water_value_function[t+1].evaluate( np.array(data['x_next']).reshape(-1,len(self.stateVariable)) )  )
        return x , y

    def get_training_set(self, t):
        if self.n_forward_pass < self.record_length:
            length_record = self.n_forward_pass + 1
        else :
            length_record = self.record_length

        x = self.state_variable_record[:length_record,t,]
        dx = self.dual_variable_record[:length_record,t,]
        # y = self.benefit_record[:length_record,t,] # current benefit
        # y += np.squeeze(self.discount_factor * self.water_value_function[t+1].evaluate(self.state_variable_record[:length_record,t+1,])) # plus expected future benefit
        n_ts = self.benefit_record[:length_record,t:,].shape[1]
        discount_vec = np.cumprod( [1.] + [self.discount_factor] * (n_ts-1) )
        y = np.sum( discount_vec * self.benefit_record[:length_record,t:,], axis = 1 )

        return x , dx, y

    def update_data_training(self, training_dict):
        self.water_value_function[self.algo_ts].data_training.extend(training_dict)

    def get_print_var(self):
        return [ i for i,var in enumerate(self.problem.variable_list) 
                    if var not in self.water_value_function[self.algo_ts].non_print_var ]


    def update_log(self):
        dict_tmp = {}
        dict_tmp['n_iter'] = self.n_forward_pass + 1
        dict_tmp['ts'] = self.algo_ts
        dict_tmp.update(self.water_value_function[self.algo_ts].get_log())
        return dict_tmp

    def get_log(self):
        return self.dict_log

    def close(self):
        for wv in self.water_value_function :
            wv.close()
