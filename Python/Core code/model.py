import math
import copy
import numpy as np
from operator import mul
from functools import reduce

MONTH_LIST = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

class Model(object):
    # parameters shared between classes
    params = type('', (), {})()
    # states of the system shared between classes

    # random variable of the system shared between classes
    random_variable = type('', (), {})()
    global_state = type('', (), {})()

    #constraints = []
    #objectives = []
    #variable_bounds = {}

    def __init__(self, modules):
        self.modules = modules
        self.params.type_vec = [module.type for module in modules]

        self.total_number_possible_sample = [1 for _ in MONTH_LIST]
        self.n_bins = [[] for _ in MONTH_LIST]
        for module in modules:
            for i in range(len(MONTH_LIST)):
                self.total_number_possible_sample[i] *= module.numberPossibleSample[i]
                self.n_bins[i].extend(module.number_possible_sample_vec[i])

    def sampleRandomProcess(self, sample_ind=[]):
        id_ini = 0
        self.total_random_sample_probability = 1
        for module in self.modules:
            id_end = id_ini + module.dim_random_process
            module.sampleRandomProcess(sample_ind[id_ini:id_end])
            id_ini += module.dim_random_process
            self.total_random_sample_probability *= module.random_sample_probability

    def get_initial_state(self, mode = 'unique'):
        return [module.get_initial_state(mode) for module in self.modules]

    def update_initial_state(self, initial_state):
        for module in self.modules:
            initial_state = module.update_initial_state(initial_state)
        return  initial_state

    def initialize(self):
        for module in self.modules:
            module.module_state.set_initial_value() #initialize()
            module.map_state_module_value()

    def set_state(self, state_value):
        for i,module in enumerate(self.modules):
            module.module_state.set_state_value(state_value[i]) #module_state.value = state_value[i]
            module.map_state_module_value()

    def set_default_state(self, t):
        for module in self.modules:
            module.module_state.set_default(t)
            module.map_state_module_value()

    def set_time_step(self, ts):
        self.global_state.time_step = ts
        self.global_state.currentMonth = self.params.timeStepProp_month[ts]
        self.global_state.currentYear = self.params.timeStepProp_year[ts]

    def get_constraints(self):
        cons = []
        for module in self.modules:
            cons.extend( module.get_constraints() )
        return cons

    def get_objectives(self):
        obj = []
        for module in self.modules:
            obj.extend( module.get_objectives() )
        return obj

    def get_var_bounds(self):
        var_bounds = {}
        for module in self.modules:
            var_bounds.update( module.get_var_bounds() )
        return var_bounds
    # def defineConstraints(self, ts):
    #     self.timeStep = 0
    #     for module in self.modules:
    #         module.defineConstraints(ts)

    # def defineObjectives(self, ts):
    #     for module in self.modules:
    #         module.defineObjectives(ts)

    # def updateValues(self):
    #     for module in self.modules:
    #         module.updateValues()

    def applyTransition(self, decision, variableList):
        for module in self.modules:
            #TODO index of state variable in variableList
            module.applyTransition(decision, variableList)

    # compute index-th random process # TODO make it an iterator
    def setIndexSampleProcess(self, index):
        self.nRandomProcess = 4  # TODO set with method
        n_bin = self.n_bins[self.global_state.currentMonth]  # TODO : vector
        residual = index
        id = np.zeros(self.nRandomProcess, dtype=int)

        for i in range(self.nRandomProcess - 1):
            id[i] = math.floor(residual / reduce(mul, n_bin[i+1:]) ) #self.nbin ** (self.nRandomProcess - i - 1))
            residual = residual - id[i] * reduce(mul, n_bin[i+1:]) #self.nbin ** (self.nRandomProcess - i - 1)
        id[self.nRandomProcess - 1] = residual

        self.sampleIndex = id
        
    def simulate(self, algo, H, file_string = ''):
        self.total_benefit_simul = 0.
        self.n_resample = 0
        discount_vec = np.cumprod( [1.] + [algo.discount_factor] * (len(MONTH_LIST)-1) )

        log_file = open('simulation_files/' + file_string  + '.txt', 'w')
        #if algo.problem == None:
        algo.make_problem()
        id_var = algo.get_print_var()
        #log_file.write( 'ts_benefit\t' + '\t'.join( [ algo.problem.variable_list[i] for i in id_var]) + "\n")

        for y in range(H):
        #unfeasible_problem = 1
        #while unfeasible_problem:
#                if y == 0:
#                    algo.set_initial_state('multi-random')
#                else:
#                    algo.set_initial_state('previous')
            algo.set_initial_state('unique')#multi-random')
            
            year_benefit = 0
            temp_log = ''
            
            for t in range(len(MONTH_LIST)):
                algo.algo_ts = t
                self.set_time_step(t)
                self.global_state.year_simul = y
                
                self.sampleRandomProcess([])
                unfeasible_problem = algo.solve_stage_problem()
                if unfeasible_problem:
                    year_benefit = 500.
                    self.n_resample += 1 
                    break
                
                ts_benef = algo.get_ts_benefit()
                year_benefit+= discount_vec[t] * ts_benef
                #temp_log += str(ts_benef) + '\t' + '\t'.join(map(str, [ algo.problem.decisions[i] for i in id_var] )) + "\n"

                self.applyTransition(algo.problem.decisions, algo.problem.variable_to_indice_dict)
            #log_file.write(temp_log)
            
            self.total_benefit_simul += year_benefit / H

        return self.total_benefit_simul
            
class Module(Model):
    def __init__(self):
        self.type = ''
        self.module_state = State([])

        self.dim_random_process = 0
        self.nbin = [[] for m in MONTH_LIST]
        self.numberPossibleSample = [1 for _ in MONTH_LIST]
        self.number_possible_sample_vec = [[] for _ in MONTH_LIST]
        self.random_sample_probability = 1

    # abstract classes
    def initialize(self):
        pass

    # def defineConstraints(self, ts):
    #     pass

    # def defineObjectives(self, ts):
    #     pass

    # def updateValues(self):
    #     pass

    def get_constraints(self):
        return []

    def get_objectives(self):
        return []

    def get_var_bounds(self):
        return {}

#    def set_state_value(self,state_value):
 #       pass

    def applyTransition(self, decision, variableList=[]):
        pass

    def sampleRandomProcess(self, sample_ind=[]):
        pass

    def map_state_module_value(self):
        pass

    def get_initial_state(self,mode):
        pass

    def update_initial_state(self, initial_state):
        return initial_state
    
    def set_historical_seq(self, sequence):
        pass

class Variable(object):
    variable_properties = [('V_TY_total_sv','V_TY_total'),
                         ('V_TY_MCA_sv','V_TY_MCA_now'),
                         ('V_NT_BC_sv','V_NT_BC_now'),
                         ('ARD_storage_sv','ARD_storage_next_ts'),
                         ('MCA_storage_sv', 'MCA_storage_next_ts'),
                         ('GMS_storage_sv', 'GMS_storage_next_ts'),
                         ('resi_sv_peace','resi_peace'),
                         ('resi_sv_columbia','resi_columbia'),
                         ('previous_forecast_residual_sv_peace','se_fc_resi_peace'),
                         ('previous_forecast_residual_sv_columbia','se_fc_resi_columbia')]
    list_variable_name_now, list_variable_name_next_ts = zip(*variable_properties)
    # TODO : split in two classes: next ts and now
    def __init__(self, variable_name):
        """ variable name is a list """
        self.n_variable = len(variable_name)
        self.name = variable_name
        self.name_next_ts = [None] * self.n_variable
        self.index = [None] * self.n_variable
        self.index_next_ts = [None] * self.n_variable

        self.set_name_next_ts()

    def set_name_next_ts(self):
        for i,n in enumerate(self.name):
            self.name_next_ts[i] = self.list_variable_name_next_ts[self.list_variable_name_now.index(n)]

    # TODO : to USE !
    def set_index(self , variableList):
        for i,n in enumerate(self.name):
            self.index[i] = variableList.index(n)
        for i,n in enumerate(self.name_next_ts):
            self.index_next_ts[i] = variableList.index(n)

# TODO change implementation variable is declared twice in module
class State(object):
    def __init__(self, state_variable=[], ini_value=[], default_value=[]):
        self.state_variable = state_variable
        self.value = [None] #if ini_value==[] else copy.copy(ini_value)
        self.ini_value = [None] if ini_value==[] else ini_value
        self.default_value = None if default_value==[] else default_value

    def set_state_value(self, _value):
        self.value = copy.copy(_value)

    def set_initial_value(self):
        self.set_state_value(self.ini_value)

    def set_default(self,ts):
        if self.default_value is not None:
            self.set_state_value(self.default_value[ts])
