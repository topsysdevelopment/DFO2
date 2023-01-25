import numpy as np
import mpi4py.MPI as MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

from algo import Algo
from bender_cuts import Bender_cuts

class SDDP(Algo):
    """docstring for sddp"""
    
    algo_name = 'SDDP'
    
    def __init__(self, model, stateVariable, solver, sample_mode):
        super(SDDP, self).__init__(model,stateVariable, solver, sample_mode)
        # declare bender cuts object
        self.water_value_type = Bender_cuts
        self.water_value_function = self.create_water_value_function()

    def backward_pass(self):
        for t in range(self.model.params.T-1, -1, -1):
            self.algo_ts = t
            self.model.set_time_step(t)
            self.model.set_state(self.stateForwardPass[t])

            slope_tot = np.zeros( len(self.stateVariable) )
            intercept_tot = 0.
            total_prob = 0.
            state_vector = np.zeros(len(self.stateVariable))
            ### loop over possible realizations
            for j in range(self.model.total_number_possible_sample[self.model.global_state.currentMonth]): # iteration over all possible random process realizations
                self.model.setIndexSampleProcess(j)  # set index j to random process for white noise generation
                self.model.sampleRandomProcess(self.model.sampleIndex)  # sample random process #j
            
                unfeasible_problem = self.solve_stage_problem()#t, 'random_process')
                if unfeasible_problem:
                    #print " **** unfeasible in backward phase **** "
                    break
                    #return 0

                # if t==11 and j== 0 :
                #     f = open("prob_t11.txt",'w')
                #     for c in self.model.get_constraints():
                #         f.write( c.__str__() + '\n')
                #     f.close()

                slope, intercept = self.return_slope_intercept()
                slope_tot = slope_tot + self.model.total_random_sample_probability * slope # compute mean slope
                intercept_tot = intercept_tot + self.model.total_random_sample_probability * intercept # compute mean intercept
                total_prob += self.model.total_random_sample_probability
            ### add cut
            #assert abs(total_prob - 1.0) < 0.01
            if unfeasible_problem:
                new_cut = None
            else:
                new_cut = self.water_value_function[t].create_cut( intercept_tot, slope_tot)
            all_cut = comm.gather(new_cut, root=0) # gather cuts from different CPU
            all_cut = comm.bcast(all_cut, root=0) # send all new cuts to all CPU
            comm.Barrier()
            #print -slope_tot*10**6
            self.water_value_function[t].add_cuts(all_cut)

        return 1

    def return_slope_intercept(self):
        state_vec = self.problem.get_decisions(self.stateVariable)
        slope = np.array( self.problem.get_dual_values(self.stateVariable))
        intercept = self.problem.objectiveValue - np.sum( slope * state_vec )
        return slope, intercept