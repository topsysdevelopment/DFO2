import numpy as np

from model import Module
from solver import linear_constraint, concave_piecewise_linear_contraint, objective


class FinalModule(Module):
    def __init__(self, system):
        super(FinalModule, self).__init__()
        self.type = 'final'

        self.system = system
        self.loadParameters()

    def get_constraints(self):#,current_month):

        t_now = self.global_state.currentMonth

        constraints = []
        # subject to V_live_cal {j in plant_storage_s} :
        #for i,plant in enumerate(self.system.plant_storage):
            #constraints.append(
            #        linear_constraint( [ 'V_live_'+ plant , plant + '_storage_next_ts'] ,
            #            [1 , -1] , 'E' , - self.params.V_Min[i] ))

        for i,h in enumerate(self.params.hplName):
            varx = 'Spot_Exp_USH_' + h
            vary = 'Spot_Exp_USH_Profit_' + h
            nparts = np.array([self.params.price_Imp_npc + self.params.price_Exp_npc])
            breakx = np.concatenate( (self.params.price_Imp_bkpt * self.params.US_tran_minH[t_now,i] , np.zeros(1), self.params.price_Exp_bkpt * self.params.US_tran_maxH[t_now,i]) )
            slope = np.concatenate((self.params.price_Imp_rate * self.params.price_Imp_USH[t_now,i] , self.params.price_Exp_rate * self.params.price_Exp_USH[t_now, i] ))
            constraints.append(
                concave_piecewise_linear_contraint(varx = varx, vary = vary, nparts = nparts, bkpointx = breakx, slope = slope) )


        constraints.append(
            linear_constraint(['Q_LCA_now'], [1], 'E', self.params.Q_LCA[t_now] ))
        
        return constraints


    def get_var_bounds(self):
        var_bounds = {}

        var_bounds['Q_LCA_now'] = 'free'
        for i,h in enumerate(self.params.hplName):
            var_bounds['Spot_Exp_USH_' + h ] = 'free'
            var_bounds['Spot_Exp_USH_Profit_' + h ] = 'free'
        return var_bounds


    def get_objectives(self):#, current_month):
        t_now = self.global_state.currentMonth
        
        objectives = []

        var =[]
        coeff=[]

        # Objective:
        # US exports
        for i,h in enumerate(self.params.hplName) :
            var.append( 'Spot_Exp_USH_Profit_' + h )
            coeff.append(self.params.HPLHR[t_now,i]* self.params.price_multiplier[i])
        
        # AB exports
        #for i,h in enumerate(self.system.hplName) :
        #    var.append('Spot_Exp_ABH_Profit_' + h)
        #    coeff.append(self.params.HPLHR[t_now,i]* self.params.price_multiplier[i])
        
        # # BCH NT release downstream benefit
        var.append('Q_NT_BC_now')
        coeff.append( self.params.HKARD[t_now] * np.sum(self.params.HPLHR[t_now]) * self.params.price_Mid_C[t_now] )
        var.append('Q_LCA_now')
        coeff.append(self.params.HKARD[t_now] * np.sum(self.params.HPLHR[t_now]) * self.params.price_Mid_C[t_now])

        # penalties
        for plant in self.system.plantName:
            for i,h in enumerate(self.params.hplName) :
                var.extend([ 'QP_min_x_' + plant + '_' + h , 'QP_max_x_' + plant + '_' + h ])
                coeff.extend( [ - 3 * self.params.HPLHR[t_now,i] * self.params.cost_x] * 2 )
            var.extend(['QS_max_x_' + plant , 'QS_min_x_' + plant ])
            coeff.extend([ - 3 * np.sum(self.params.HPLHR[t_now]) * self.params.cost_x] * 2)

        for i,h in enumerate(self.params.hplName) :
            var.append('load_x_' + h )
            coeff.append( - self.params.HPLHR[t_now,i] * self.params.cost_x     )

        var.extend(['Q_NT_BC_min_x' , 'Q_NT_BC_max_x'])
        coeff.extend([- 2 * 3 * np.sum(self.params.HPLHR[t_now]) * self.params.cost_x ] * 2 )

        coeff_scale = [ - c/10**6 for c in coeff]
        objectives.append(objective(coeff_scale, var))

        return objectives

    def loadParameters(self):
        self.params.cost_x = 113.
