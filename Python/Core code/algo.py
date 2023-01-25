import numpy as np
import copy
import mpi4py.MPI as MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

from solver import Linear_problem, objective, linear_constraint
from model import Variable
from water_value import WvFunction

class Algo(object):
    """docstring for Algo"""

    def __init__(self, model , stateVariable, solver_name, sample_mode):
        self.model = model
        self.stateVariable = stateVariable
        self.state_variable_next_ts = [ 
            Variable.list_variable_name_next_ts[Variable.list_variable_name_now.index(var)] 
                                      for var in stateVariable ]
        self.discount_factor = model.params.discount
        self.discount_vec = np.cumprod( [1.] + [self.discount_factor] * (self.model.params.T-1) )
        self.n_fixed_fwd_pass = 1 

        self.wv_objective = [objective([self.discount_factor],['water_value'])]
        self.null_wvf_cons = [linear_constraint( ['water_value'] , [1.] , 'G' , 0. )]
        self.wv_variable_bounds = {'water_value' : 'free' }

        self.end_wv = False
        
        self.stateForwardPass = [ [ [None] for _ in range(len(self.model.modules))] for _ in range(self.model.params.T+1) ]
        
        self.forward_benefit = np.zeros(self.model.params.T)
        self.forward_state_variable = np.zeros( (self.model.params.T+1, len(stateVariable) ) )
        self.forward_dual_variable = np.zeros( (self.model.params.T, len(stateVariable) ) )

        self.inialization_mode = sample_mode #'unique'#'multi-random'
        self.var_HK = False

        self.record_length = 100

        self.benefit_record = np.zeros( (self.record_length, self.model.params.T) )
        self.state_variable_record = np.zeros( (self.record_length, self.model.params.T+1, len(stateVariable) ) )
        self.dual_variable_record = np.zeros( (self.record_length, self.model.params.T, len(stateVariable) ) )
        self.first_stage_benefit = 0.

        self.solver_name = solver_name
        self.problem = Linear_problem(self.solver_name) 
        self.algo_ts = 0
        
        self.offset = 0

        self.dict_log = []

    # initialize system
    def set_initial_state(self, type = 'unique'):
        if type == 'unique':
            initial_state = self.model.get_initial_state()
        elif type == 'multi_random': 
            initial_state = self.model.get_initial_state('multi_random')
        elif type == 'previous':
            initial_state = self.stateForwardPass[-1]
        else:
            print('Initialization method not implemented')

        initial_state = self.model.update_initial_state(initial_state)
        self.model.set_state(initial_state)

    def solve_stage_problem(self):#, t, update_case=[] ):
        # 1st solve : average decision
        self.make_problem()

        unfeasible_problem = self.problem.solve()

        if unfeasible_problem:
            # print 'Re-sample forward scenario', 'rank = %d' % rank
            return unfeasible_problem

        #2nd solve : update end storage
  
        # TODO :
        # if self.var_HK is True:
        #     self.model.modules[3].updateValues(
        #         [self.problem[t].decisions[self.problem[t].variable_to_indice_dict[var]] for var in
        #          self.model.modules[3].recall_var])  # update HK
        #     self.problem[t].update_constraint_values("var_HK")  # update value in the solver
        #     unfeasible_problem = self.problem[t].solve()  # solve stagewise problem

        return unfeasible_problem

    def make_problem(self):
        cons = self.model.get_constraints()
        obj = self.model.get_objectives()
        var_bounds = self.model.get_var_bounds()
        
        wv_cons = self.get_wv_constraints()
        if wv_cons == []:
            wv_cons = self.null_wvf_cons
        cons.extend( wv_cons  )
        obj.extend( self.wv_objective )
        var_bounds.update( self.get_wv_variable_bounds() )

        self.problem.populate_problem( obj, cons, var_bounds )

    def get_ts_benefit(self):
        total_benefit = self.problem.objectiveValue
        future_value = self.problem.get_decisions(['water_value'])[0]
        return total_benefit - self.discount_factor * future_value 

    def create_water_value_function(self):
        wv_list = [ self.water_value_type(self.state_variable_next_ts, name = "time_step_" + str(t)) for t in range(self.model.params.T) ]
        default = WvFunction()
        wv_list.append( default )
        return wv_list

    def get_wv_constraints(self):
        return self.water_value_function[self.algo_ts+1].get_constraints()

    def get_wv_variable_bounds(self):
        return self.wv_variable_bounds

    def update_end_wvf(self):
        # delete all bender cuts !
        self.water_value_function[-1].copy(self.water_value_function[0])

        for wv_func in self.water_value_function[:-1]:
            wv_func.reset_wv()
            
    def save_water_value(self):
        for t, wv_func in enumerate(self.water_value_function):
            wv_func.save(str(t))
           
    def load_water_value(self):
        for t, wv_func in enumerate(self.water_value_function):
            wv_func = wv_func.load(str(t))

    def get_print_var(self):
        return range(len(self.problem.variable_list))

#class forward_backward_algo(Algo):
#    def _init_(self, model, stateVariable, solver, sample_mode):
#        super(forward_backward_algo, self).__init__(model,stateVariable, solver, sample_mode)

    def optimize(self, nCuts, n_simulate = None):
        for n_cut in range(nCuts):  # loop over cuts
            # run forward and backward phase
            self.forward_backward_pass(n_cut + self.offset)
            # simulate the system with the current cuts
            if n_simulate is not None:
                if (n_cut+1)% n_simulate == 0 :
                    print(self.model.simulate( self, 1000, '' ))


    def forward_backward_pass(self, n_cut):

#        backward_success_generation = 0

#        while backward_success_generation != 1:
#            forward_success_generation = 0
#            while forward_success_generation != 1:
        # initialize state
        self.set_initial_state(self.inialization_mode)
        # forward loop
        forward_success_generation = self.forward_pass(n_cut)

        # backward loop
        #self.dict_log = []
        backward_success_generation = self.backward_pass()
        #print rank
        #comm.Barrier()

        id_record = n_cut % self.record_length
        self.benefit_record[id_record] = self.forward_benefit
        self.state_variable_record[id_record] = self.forward_state_variable
        self.dual_variable_record[id_record] = self.forward_dual_variable
        
        #if rank ==0 and n_cut >= self.n_fixed_fwd_pass:
        #    discount_benef = np.dot( self.discount_vec , self.forward_benefit )
        #    print('%d\t%f' % ( n_cut, discount_benef) ) 
        
    def forward_pass(self,n):
        self.n_forward_pass = n
        success_generation = self.forward_pass_core(n)
        return success_generation
    
    def forward_pass_core(self,n):
        for t in range(self.model.params.T):
            self.algo_ts = t
            self.model.set_time_step(t)

            self.model.sampleRandomProcess([])  # sample random process

            if n < self.n_fixed_fwd_pass: # TODO: Pass as parameter
                self.model.set_default_state(self.model.global_state.currentMonth-1)

            unfeasible_problem = self.solve_stage_problem()
            if unfeasible_problem:
                return 0
                # self.record_benefit_nan(t)
                # self.record_state_variable_nan(t)
                # return 1
            if t == 0 and rank == 0 : 
                self.first_stage_benefit = self.problem.objectiveValue 
                #print('Rank = %d, Cut = %d, Total benefit = %f' % (rank, n, self.problem.objectiveValue) ) 
                

            self.recordStateForwardPass(t)
            self.record_benefit(t)
            self.record_state_variable(t)
            self.record_dual_variable(t)
            self.model.applyTransition(self.problem.decisions, self.problem.variable_to_indice_dict)

        if n < self.n_fixed_fwd_pass:  
            self.model.set_time_step(self.model.params.T%12)
            self.model.set_default_state(self.model.global_state.currentMonth - 1)

        self.recordStateForwardPass(self.model.params.T)
        self.record_state_variable(self.model.params.T)
        
        return 1

    # record the evolution of the system in stateForwardPass list
    def recordStateForwardPass(self,t):
        for i,module in enumerate(self.model.modules):
            self.stateForwardPass[t][i] = copy.copy(module.module_state.value)

    # record the time step benefit in the forward pass
    def record_benefit(self,t):
        self.forward_benefit[t] = self.get_ts_benefit()

    def record_dual_variable(self,t):
        self.forward_dual_variable[t] = self.problem.get_dual_values(self.stateVariable)

    # record the state variable valuein the forward pass
    def record_state_variable(self,t):
        if t == self.model.params.T :
            self.forward_state_variable[t] = self.problem.get_decisions(self.state_variable_next_ts)
        else:
            self.forward_state_variable[t] = self.problem.get_decisions(self.stateVariable)

    def close(self):
        pass
        
    # def record_benefit_nan(self,t):
    #     self.forward_benefit[t:] = 1000.

    # def record_state_variable_nan(self,t):
    #     self.forward_state_variable[t:] = np.nan