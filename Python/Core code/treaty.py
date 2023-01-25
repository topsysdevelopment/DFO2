from readFileTest import *
from solver import linear_constraint
from model import Module, State, Variable

# TODO : in utils class
MONTH_LIST = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
dayPerMonth = [31,28,31,30,31,30,31,31,30,31,30,31]

class Treaty(Module):
    """docstring for Treaty"""
    def __init__(self,path,system):
        super(Treaty,self).__init__()
        self.type = 'treaty'

        self.system = system 
        self.fccPlantName = self.system.fccPlantName
        self.hplName = self.params.hplName
        self.HPLHR = self.params.HPLHR
        ### FCC
        self.fcc_shift = np.array([885137, 913690 ]) #[MCA ARD]
        ### Storage accounts
        # total treaty storage
        self.V_TY_total_ini = 80000	#201297.38
        self.V_TY_sys_ini  = 201297.38
        # TY 
        self.V_TY_MCA_ini = 40932#99934.87
        self.V_TY_MCA_max = 128570
        self.V_TY_MCA_min = -36703
        self.V_TY_ARD_max = 101363
        # NT
        self.V_NT_BC_ini = 28000 #18963
        # MCA and ARD
        self.params.V_MCA_recall = 0. # or 7440.11 ?

        self.loadParameters(path , self.fccPlantName)

        self.stage_se_fc_Jul = ['Jan','Feb','Mar','Apr','May','Jun'] 
        self.params.fcc_stage = ['Oct','Nov','Dec','Jan','Feb','Mar','Apr' ]
        self.fcc_sto_stage = ['Jan','Feb','Mar','Apr']  
        self.fcc_fix_stage = [x for x in self.params.fcc_stage if x not in self.fcc_sto_stage ]
        self.sf_TD_stage = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug']

        self.constraint_ARD_TTY_release_cal_var = ['Q_TY_ARD_now', 'V_TY_total' , 'V_TY_total_sv' ]
        self.constraint_ARD_TTY_release_cal_coeff = [-99.,-99.,-99.]
        self.constraint_ARD_TTY_release_cal_rhs = [-99.]
        self.constraints_TTY_composite_storage_cal_var = ['V_TY_total', 'V_TY_sys']
        self.constraints_TTY_composite_storage_cal_coeff = [-99.,-99.]
        self.constraints_TTY_composite_storage_cal_rhs = [-99.]
        self.constraint_TTY_system_storage_cal_var = ['V_TY_sys', 'V_TY_total_sv']
        self.constraint_TTY_system_storage_cal_coeff = [-99.,-99.]
        self.constraint_TTY_system_storage_cal_rhs = [-99.]
        #self.V_TY_total_sv_rhs = [-99.]
        #self.V_NT_BC_sv_rhs = [-99.]
        #self.V_TY_MCA_sv_rhs = [-99.]
        #self.treaty_mass_balance_rhs = [-99.]
        #self.vmax_mca_rhs = [-99.]
        #self.vmax_ard_rhs = [-99.]
        self.load_resource_balance_rhs = np.zeros(len(self.hplName))

        self.fcc_1 = np.zeros( 2 )
        self.above_threshold = 1
        self.V_TY_MCA_value, self.V_TY_total_value, self.V_NT_BC_value = 0., 0., 0.
        state_variable = Variable(['V_TY_MCA_sv','V_TY_total_sv','V_NT_BC_sv'])
        self.ini_value = np.array([self.V_TY_MCA_ini,self.V_TY_total_ini,self.V_NT_BC_ini])
        self.module_state = State(state_variable, self.ini_value, self.params.V_treaty_select) # TODO : default value

        #self.initialize()
        #self.updateValues()

#    def initialize(self):
#        self.V_TY_MCA_value, self.V_TY_total_value, self.V_NT_BC_value = self.module_state.ini_value
    def get_initial_state(self, mode):
        if mode == 'unique':
            return self.ini_value
        elif mode == 'multi_random':
            return [np.random.choice(val) for val in self.params.treaty_multi_ini.T]
        else:
            print('initialization mode not managed')

    def map_state_module_value(self):
        self.V_TY_MCA_value, self.V_TY_total_value, self.V_NT_BC_value = self.module_state.value
        self.global_state.V_TY_MCA_value, self.global_state.V_TY_total_value, self.global_state.V_NT_BC_value = self.V_TY_MCA_value, self.V_TY_total_value, self.V_NT_BC_value

    def applyTransition(self, decision, variable_list_dict):
        for i in range(3):#self.module_state.len):
            var_next_ts = self.module_state.state_variable.name_next_ts[i]
            self.module_state.value[i] = decision[variable_list_dict[var_next_ts]]
        self.V_TY_MCA_value, self.V_TY_total_value, self.V_NT_BC_value = self.module_state.value
        self.global_state.V_TY_MCA_value, self.global_state.V_TY_total_value, self.global_state.V_NT_BC_value = self.V_TY_MCA_value, self.V_TY_total_value, self.V_NT_BC_value

    def updateValues(self):
        ''' Upadte the TSR equations. Must be called after the inflow generation'''
        t_now = self.global_state.currentMonth

        sf_TD = self.params.sf_TD_intercept[t_now] + self.params.sf_TD_slope[t_now] * self.system.random_variable.se_fc[1] if MONTH_LIST[t_now] in self.sf_TD_stage else 0.
        sa_TD = self.params.sa_TD_intercept[t_now] + self.params.sa_TD_slope[t_now] * sf_TD if MONTH_LIST[t_now] in self.sf_TD_stage else 0.

        self.fcc_1 = self.params.fcc_min[t_now] if MONTH_LIST[t_now] in self.fcc_fix_stage else (sf_TD-self.fcc_shift) * self.params.fcc_slope[t_now] + self.params.fcc_intercept[t_now] if MONTH_LIST[t_now] in self.fcc_sto_stage else 0.
        # make sure fcc_1 is between fcc_min and fcc_max
        self.fcc_corrected = np.zeros(len(self.fccPlantName))
        for j in range(len(self.fccPlantName)):
            if MONTH_LIST[t_now] in self.fcc_sto_stage:
                if self.params.fcc_max[t_now,j]< self.fcc_1[j]:
                    self.fcc_corrected[j] = self.params.fcc_max[t_now,j]- self.fcc_1[j]
                elif self.fcc_1[j] < self.params.fcc_min[t_now,j]:
                    self.fcc_corrected[j] = self.params.fcc_min[t_now,j]- self.fcc_1[j]
            else:
                self.fcc_corrected[j] = 0.

        self.fcc_1 = self.fcc_1 + self.fcc_corrected
        self.global_state.fcc_1 = self.fcc_1

        # total treaty storage and ARD treaty outflow
        self.V_TY_total_corrected = 0.
        self.Q_TY_ARD_corrected = 0.

        TD_inflow = self.params.TD_inflow_intercept[t_now] + self.params.TD_inflow_slope[t_now] * self.random_variable.riverInflows[1]
        CR_se_fc_Jul = self.params.CR_se_fc_Jul_intercept[t_now] + self.params.CR_se_fc_Jul_slope[t_now] * self.random_variable.se_fc[1] if MONTH_LIST[t_now] in self.stage_se_fc_Jul else 0

        TD_se_fc_Jul = self.params.TD_se_fc_Jul_intercept[t_now] + self.params.TD_se_fc_Jul_slope[t_now] * CR_se_fc_Jul if MONTH_LIST[t_now] in self.stage_se_fc_Jul else 0


        if self.V_TY_total_value > self.params.D1[t_now]:
            self.above_threshold = 1
        elif self.random_variable.riverInflows[1] > self.params.D2[t_now]:
            self.above_threshold = 1
        elif TD_inflow - self.random_variable.riverInflows[1] > self.params.D3[t_now] :
            self.above_threshold = 1
        elif MONTH_LIST[t_now] in self.stage_se_fc_Jul :
            if CR_se_fc_Jul - self.random_variable.riverInflows[1]*dayPerMonth[t_now] > self.params.D4[t_now] :
                self.above_threshold = 1
            elif TD_se_fc_Jul - TD_inflow*dayPerMonth[t_now] - (CR_se_fc_Jul - self.random_variable.riverInflows[1]*dayPerMonth[t_now]) > self.params.D5[t_now]:
                self.above_threshold = 1
            else:
                self.above_threshold = 0
        else:
            self.above_threshold = 0

        # From here it needs constraints to get V_TY_total_value as sv
        # V_TY_sys
        if self.above_threshold == 0 :
            V_TY_sys = self.params.X0_1[t_now] + self.params.X1_1[t_now] * self.V_TY_total_value \
                       + self.params.X2_1[t_now] * self.random_variable.riverInflows[1] + self.params.X3_1[t_now] * ( TD_inflow - self.random_variable.riverInflows[1] )
            if MONTH_LIST[t_now] in self.stage_se_fc_Jul :
                V_TY_sys = V_TY_sys + self.params.X4_1[t_now] * (CR_se_fc_Jul-self.random_variable.riverInflows[0]*dayPerMonth[t_now]) \
                            + self.params.X5_1[t_now] * ( TD_se_fc_Jul-TD_inflow*dayPerMonth[t_now] - (CR_se_fc_Jul-self.random_variable.riverInflows[1]*dayPerMonth[t_now]) )
        else :
            V_TY_sys = self.params.X0_2[t_now] + self.params.X1_2[t_now] * self.V_TY_total_value \
                       + self.params.X2_2[t_now] * self.random_variable.riverInflows[1] + self.params.X3_2[t_now] * ( TD_inflow - self.random_variable.riverInflows[1] )
            if MONTH_LIST[t_now] in self.stage_se_fc_Jul :
                V_TY_sys = V_TY_sys + self.params.X4_2[t_now] * (CR_se_fc_Jul - self.random_variable.riverInflows[1]*dayPerMonth[t_now]) \
                            + self.params.X5_2[t_now] * ( TD_se_fc_Jul-TD_inflow*dayPerMonth[t_now] - (CR_se_fc_Jul-self.random_variable.riverInflows[1]*dayPerMonth[t_now]) )

        if V_TY_sys > (self.params.V_TY_total_max[t_now] - max(0, self.params.V_SOA[t_now]+ self.params.V_LCA[t_now]) ) :  
               self.V_TY_total_corrected = self.params.V_TY_total_max[t_now] - max(0, self.params.V_SOA[t_now]+ self.params.V_LCA[t_now]) - V_TY_sys 
        elif V_TY_sys  < self.params.V_TY_total_min[t_now] :
            self.V_TY_total_corrected = self.params.V_TY_total_min[t_now] - V_TY_sys
        else :
            self.V_TY_total_corrected = 0.


        self.V_TY_total = V_TY_sys + self.V_TY_total_corrected
        self.Q_TY_ARD_now = self.V_TY_total_value / dayPerMonth[t_now] + self.random_variable.riverInflows[1] - self.V_TY_total / dayPerMonth[t_now]
        self.Q_TY_ARD_corrected = 0. if self.Q_TY_ARD_now >= 0. else - self.Q_TY_ARD_now
        self.Q_TY_ARD_now = self.Q_TY_ARD_now if self.Q_TY_ARD_corrected <= 0. else 0.
        self.V_TY_total = V_TY_sys + self.V_TY_total_corrected - self.Q_TY_ARD_corrected * dayPerMonth[t_now]

        

        #TTY_system_storage_cal
        if self.above_threshold == 0 :
            X0,X1,X2,X3,X4,X5 = self.params.X0_1[t_now],self.params.X1_1[t_now],self.params.X2_1[t_now],self.params.X3_1[t_now],self.params.X4_1[t_now],self.params.X5_1[t_now]
        else:
            X0,X1,X2,X3,X4,X5 = self.params.X0_2[t_now],self.params.X1_2[t_now],self.params.X2_2[t_now],self.params.X3_2[t_now],self.params.X4_2[t_now],self.params.X5_2[t_now]
        coeff = [ 1 , - X1 ]
        rhs = X0 + X2 * self.random_variable.riverInflows[1] + X3 * ( TD_inflow - self.random_variable.riverInflows[1] )
        if MONTH_LIST[t_now] in self.stage_se_fc_Jul :
            rhs = rhs + X4 * (CR_se_fc_Jul-self.random_variable.riverInflows[1]*dayPerMonth[t_now]) + X5 * ( TD_se_fc_Jul - TD_inflow*dayPerMonth[t_now] - (CR_se_fc_Jul-self.random_variable.riverInflows[1]*dayPerMonth[t_now]) )
        self.constraint_TTY_system_storage_cal_var[:] = ['V_TY_sys' , 'V_TY_total_sv']
        self.constraint_TTY_system_storage_cal_coeff[:] = coeff
        self.constraint_TTY_system_storage_cal_rhs[:] = [rhs]

        #self.V_TY_total_sv_rhs[:] = [self.V_TY_total_value ]
        #self.V_NT_BC_sv_rhs[:] = [self.V_NT_BC_value ]
        #self.V_TY_MCA_sv_rhs[:] = [self.V_TY_MCA_value ]
        #self.treaty_mass_balance_rhs[:] = [self.random_variable.inflows[self.system.plantName.index('MCA')] * dayPerMonth[t_now] ]
        #self.vmax_mca_rhs[:] = self.params.VMax1[self.system.plantName.index('MCA')] - self.fcc_1[0]
        #self.vmax_ard_rhs[:] = self.params.VMax1[self.system.plantName.index('ARD')] - self.fcc_1[1]

        for i in range(len(self.hplName)):
            POTHH_SH_val = self.params.POTHH_SH_intercept[t_now, i] + self.params.POTHH_SH_slope[t_now, i] * np.sum(
                self.random_variable.forecast)
            self.load_resource_balance_rhs[i] = - self.params.P_Imports[t_now, i] + self.params.P_Exports[t_now, i] + \
                                                self.params.LOADH[t_now, i] - self.params.POTHH_non_SH[t_now, i] - \
                                                self.params.NT_LOSS[t_now]


    def get_constraints(self):#, current_month):
        t_now = self.global_state.currentMonth
        month_name = MONTH_LIST[self.global_state.currentMonth]
        self.updateValues()

        constraints = []
        ### FCC
        if month_name in self.sf_TD_stage :
            constraints.append(
                linear_constraint( ['sf_TD', 'se_fc_columbia' ], [1., - self.params.sf_TD_slope[t_now] ], "E", self.params.sf_TD_intercept[t_now] ) )
        else:
            constraints.append(
                linear_constraint( ['sf_TD'], [1.], 'E' , 0. ) )

        if month_name in self.sf_TD_stage :
            constraints.append(
                linear_constraint( ['sa_TD', 'sf_TD' ], [1., - self.params.sa_TD_slope[t_now] ], "E", self.params.sa_TD_intercept[t_now] ) )
        else:
            constraints.append(
                linear_constraint( ['sa_TD'], [1.], "E", 0. ) )


        for j, plant in enumerate(self.fccPlantName):
            if month_name not in self.params.fcc_stage:
                l = linear_constraint( [ "fcc1_" + plant ] , [1.] ,'E', 0.  )
            else : 
                if month_name in self.fcc_fix_stage :
                    l = linear_constraint( [ "fcc1_" + plant ] , [1.] ,'E', self.params.fcc_min[t_now,j] )
                else : 
                     if month_name in self.fcc_sto_stage :
                        if self.fcc_corrected[j] < 0 :
                            l = linear_constraint( [ "fcc1_" + plant ] , [1.] ,'E', self.params.fcc_max[t_now,j] )
                        elif self.fcc_corrected[j] == 0 : 
                            l = linear_constraint( [ "fcc1_" + plant, 'sf_TD' ] , [1., - self.params.fcc_slope[t_now,j] ] ,'E', -self.fcc_shift[j] * self.params.fcc_slope[t_now,j] + self.params.fcc_intercept[t_now,j]  )
                        else:
                            l = linear_constraint( [ "fcc1_" + plant ] , [1.] ,'E', self.params.fcc_min[t_now,j] )
            constraints.append(l)               

        ### Storage accounts
        # total treaty storage
        # TODO : dynamic update of V_TY_total
        constraints.append( 
                    linear_constraint( [ 'V_TY_total_sv' ] , [1.], 'E', self.V_TY_total_value , state_variable='V_TY_total_sv') )
        #TTY_composite_storage_cal
        # ARD_TTY_release_cal :


        #TTY_composite_storage_cal
        if self.V_TY_total_corrected >= 0. :
            constraints.append(
                linear_constraint(['V_TY_total' , 'V_TY_sys'], [1 , -1], 'E', self.V_TY_total_corrected - self.Q_TY_ARD_corrected * dayPerMonth[t_now]))
        else:
            constraints.append(
                linear_constraint(['V_TY_total' ], [1 ], 'E', self.params.V_TY_total_max[t_now] - max(0. , self.params.V_SOA[t_now] + self.params.V_LCA[t_now]) - self.Q_TY_ARD_corrected * dayPerMonth[t_now] ))
      
        # ARD_TTY_release_cal :
        if self.Q_TY_ARD_corrected <= 0. :
            constraints.append(
                linear_constraint( ['Q_TY_ARD_now', 'V_TY_total' , 'V_TY_total_sv' , 'Q_river_columbia' ], 
                    [1, 1. / dayPerMonth[t_now], - 1. / dayPerMonth[t_now], -1.], 'E', 0. ))
        else:
            constraints.append(
                linear_constraint( ['Q_TY_ARD_now'] , [1.], 'E' , 0. ) )


        # Mid-Columbia River and Snake River to calculate TDA monthly inflow
        constraints.append(
                linear_constraint( ['MC_inflow', 'Q_river_columbia'] , [1. , - self.params.MC_inflow_slope[t_now] ],'E', self.params.MC_inflow_intercept[t_now] ))

        constraints.append(
                linear_constraint( ['SR_inflow', 'MC_inflow'] , [ 1. , - self.params.SR_inflow_slope[t_now] ],'E', self.params.SR_inflow_intercept[t_now] ))

        constraints.append(
                linear_constraint( ['TD_inflow', 'Q_river_columbia'] , [ 1. , - self.params.TD_inflow_slope[t_now] ],'E', self.params.TD_inflow_intercept[t_now] ))

        if month_name in self.stage_se_fc_Jul :
            constraints.append(
                linear_constraint( ['CR_se_fc_Jul', 'se_fc_columbia'] , [1., - self.params.CR_se_fc_Jul_slope[t_now] ],'E', self.params.CR_se_fc_Jul_intercept[t_now] ))
        else :
            constraints.append(
                linear_constraint( ['CR_se_fc_Jul'] , [1. ],'E', 0. ))
    
        if month_name in self.stage_se_fc_Jul :
            constraints.append(
                linear_constraint( ['MC_se_fc_Jul', 'CR_se_fc_Jul'] , [1., - self.params.MC_se_fc_Jul_slope[t_now] ],'E', self.params.MC_se_fc_Jul_intercept[t_now] ))
        else :
            constraints.append(
                linear_constraint( ['MC_se_fc_Jul'] , [1. ],'E', 0. ))


        if month_name in self.stage_se_fc_Jul :
            constraints.append(
                linear_constraint( ['SR_se_fc_Jul', 'MC_se_fc_Jul'] , [1., - self.params.SR_se_fc_Jul_slope[t_now] ],'E', self.params.SR_se_fc_Jul_intercept[t_now] ))
        else :
            constraints.append(
                linear_constraint( ['SR_se_fc_Jul'] , [1. ],'E', 0. ))


        # if Mid-Columbia River and Snake River are not used
        if month_name in self.stage_se_fc_Jul :
            constraints.append(
                linear_constraint( ['TD_se_fc_Jul', 'CR_se_fc_Jul'] , [1., - self.params.TD_se_fc_Jul_slope[t_now] ],'E', self.params.TD_se_fc_Jul_intercept[t_now] ))
        else :
            constraints.append(
                linear_constraint( ['TD_se_fc_Jul'] , [1. ],'E', 0. ))


        #TTY_system_storage_cal : 
        var = ['V_TY_sys']
        coeff = [1. ] 
        rhs = 0.
        if self.above_threshold == 0. :
            rhs += self.params.X0_1[t_now]
            var += ['V_TY_total_sv']
            coeff += [ -self.params.X1_1[t_now] ]
            if month_name in self.stage_se_fc_Jul :
                var += ['CR_se_fc_Jul', 'Q_river_columbia', 'TD_se_fc_Jul', 'TD_inflow']
                coeff+= [ - self.params.X4_1[t_now] + self.params.X5_1[t_now], - self.params.X2_1[t_now] + self.params.X3_1[t_now] +  self.params.X4_1[t_now] *dayPerMonth[t_now] -self.params.X5_1[t_now] *dayPerMonth[t_now], - self.params.X5_1[t_now] , - self.params.X3_1[t_now] + self.params.X5_1[t_now]*dayPerMonth[t_now]]
            else :
                var += ['Q_river_columbia', 'TD_inflow']
                coeff += [ - self.params.X2_1[t_now] + self.params.X3_1[t_now] , - self.params.X3_1[t_now] ]
        else :
            rhs += self.params.X0_2[t_now]
            var += ['V_TY_total_sv']
            coeff += [ -self.params.X1_2[t_now] ]
            if month_name in self.stage_se_fc_Jul :
                var += ['CR_se_fc_Jul', 'Q_river_columbia', 'TD_se_fc_Jul', 'TD_inflow']
                coeff+= [ - self.params.X4_2[t_now] + self.params.X5_2[t_now], - self.params.X2_2[t_now] + self.params.X3_2[t_now] +  self.params.X4_2[t_now] *dayPerMonth[t_now] -self.params.X5_2[t_now] *dayPerMonth[t_now], - self.params.X5_2[t_now] , - self.params.X3_2[t_now] + self.params.X5_2[t_now]*dayPerMonth[t_now]]
            else :
                var += ['Q_river_columbia', 'TD_inflow']
                coeff += [ - self.params.X2_2[t_now] + self.params.X3_2[t_now] , - self.params.X3_2[t_now] ]
        constraints.append(
                        linear_constraint( var, coeff, 'E' , rhs ))  

        # SOA and LCA
        # TY 
        constraints.append( 
                    linear_constraint( [ 'V_TY_MCA_sv' ] , [ 1 ] , 'E', self.V_TY_MCA_value , state_variable='V_TY_MCA_sv') )
        constraints.append(
                    linear_constraint( [ 'V_TY_MCA_now' , 'V_TY_MCA_sv' , 'Q_reservoir_MCA' ] + [ 'Q_TY_MCA_now_' + x for x in self.hplName ] ,
                                    [ 1. , -1., - dayPerMonth[t_now] ] + [ float(x) /24  for x in self.HPLHR[t_now] ] , 'E', 0. ) )
        constraints.append(
                    linear_constraint( [ 'V_TY_MCA_now' , 'V_TY_ARD_now', 'V_TY_total' ] , [ 1 , 1 , -1 ] , 'E', 0. ) )
        constraints.append(
                    linear_constraint( [ 'V_TY_MCA_now'  ] , [ 1  ] , 'L', self.params.V_TY_max[t_now,0] ) )
        constraints.append(
                    linear_constraint( [ 'V_TY_ARD_now'  ] , [ 1  ] , 'L', self.params.V_TY_max[t_now,1] ) )
        constraints.append(
                    linear_constraint( [ 'V_TY_MCA_now'  ] , [ 1  ] , 'G', self.V_TY_MCA_min ) )

        
        # NT
        constraints.append( 
                            linear_constraint( [ 'V_NT_BC_sv'] , [ 1  ] , 'E', self.V_NT_BC_value , state_variable='V_NT_BC_sv' ) )
        constraints.append(
                            linear_constraint( [ 'V_NT_BC_now', 'V_NT_BC_sv' , 'Q_NT_BC_now'  ] , [ 1. , -1. , dayPerMonth[t_now] ] , 'E', 0. ) )
        constraints.append(
                            linear_constraint( [ 'V_NT_BC_now' ] , [ 1. ] , 'G', self.params.V_NT_BC_min[t_now] ) )
        constraints.append(
                            linear_constraint( [ 'V_NT_BC_now' ] , [ 1. ] , 'L', self.params.V_NT_BC_max[t_now] ) ) 
        constraints.append( 
                            linear_constraint( [ 'Q_NT_BC_now', 'Q_NT_BC_min_x' ] , [ 1. , 1. ] , 'G', self.params.Q_NT_BC_min[t_now] ) )
        constraints.append( 
                            linear_constraint( [ 'Q_NT_BC_now', 'Q_NT_BC_max_x' ] , [ 1. , -1. ] , 'L', self.params.Q_NT_BC_max[t_now] ) )
        constraints.append( 
                            linear_constraint( [ 'Q_NT_BC_min_x' ] , [ 1. ] , 'G', 0. ) )
        constraints.append( 
                            linear_constraint( [ 'Q_NT_BC_max_x' ] , [ 1. ] , 'G', 0. ) )

        
        constraints.append(
                            linear_constraint( [ 'V_NT_BC_now', 'V_NT_US_now' ] , [ 1. , -1. ] , 'E', 0. ) )
        constraints.append(
                            linear_constraint( [ 'Q_NT_BC_now', 'Q_NT_US_now' ] , [ 1. , -1. ] , 'E', 0. ) ) 

        # MCA and ARD
        #constraints.append(
        #    linear_constraint(['V_TY_MCA_now', 'V_NT_BC_now', 'V_NT_US_now'], [1, 1, 1], 'G',
        #                      - self.params.V_SOA[t_now] - self.V_MCA_recall))
        constraints.append(
            linear_constraint(['V_TY_MCA_now', 'V_NT_BC_now', 'V_NT_US_now', 'MCA_storage_next_ts'], [1, 1, 1, -1], 'E',
                              - self.params.V_SOA[t_now] - self.params.V_MCA_recall - self.params.VMin1[
                                  self.system.plantName.index('MCA')]))

        for h in self.hplName:
            constraints.append(
                    linear_constraint( [ 'Q_TY_MCA_now_'+ h , 'Q_NT_BC_now', 'Q_NT_US_now', 'QP_MCA_' + h], [ 1, 1, 1, -1], 'E',  - self.params.Q_SOA[t_now] ))

        constraints.append(
                    linear_constraint( [ 'V_TY_ARD_now', 'ARD_storage_next_ts' ] , [ 1 , -1 ] , 'E', - self.params.V_LCA[t_now] - self.params.VMin1[self.system.plantName.index('ARD')] ) )

        for h in self.hplName:
            constraints.append(
                    linear_constraint( [ 'Q_TY_ARD_now', 'Q_NT_BC_now', 'Q_NT_US_now', 'QP_ARD_' + h] , [ 1 , 1 , 1 , -1 ] , 'E',  - self.params.Q_SOA[t_now]  - self.params.Q_LCA[t_now] ) )       
        

        for j, plant in enumerate(self.fccPlantName):
            if month_name in self.params.fcc_stage:
                constraints.append(
                    linear_constraint( [ plant +'_storage_next_ts', 'fcc1_' + plant] , [ 1. , 1. ] , 'L', self.params.VMax1[self.system.plantName.index(plant)] ) )
            else:
                constraints.append(
                    linear_constraint( [ plant +'_storage_next_ts'] , [ 1. ] , 'L', self.params.VMax1[self.system.plantName.index(plant)] ) )
            

        #constraints.append(
        #            linear_constraint( [ 'MCA_storage_next_ts'] , [ 1  ] , 'L', self.vmax_mca_rhs , is_dynamic={'cases':["random_process"],'sides':'rhs'} ) )
        
        #constraints.append(
        #            linear_constraint( [ 'ARD_storage_next_ts'] , [ 1  ] , 'L', self.vmax_ard_rhs , is_dynamic={'cases':["random_process"],'sides':'rhs'} ) )
        
        constraints.append(
                    linear_constraint( [ 'MCA_storage_average' , 'MCA_storage_next_ts' , 'V_TY_MCA_sv' , 'V_NT_BC_sv' ],
                        [ 1 , -1./2 , -1./2 , -1. ], 'E' , self.params.V_SOA[t_now-1] /2 + self.params.V_MCA_recall/2 + self.params.VMin1[self.system.plantName.index('MCA')]/2 ) )
        constraints.append(
                    linear_constraint( ['ARD_storage_average',  'ARD_storage_next_ts', 'V_TY_total_sv', 'V_TY_MCA_sv'],
                        [1 , -1./2, -1./2, 1./2], 'E', (self.params.V_LCA[t_now-1] + self.params.VMin1[self.system.plantName.index('ARD')])/2 ) )

        for i,h in enumerate(self.hplName):
                constraints.append(
                    linear_constraint(['production_' + x + '_' + h for x in self.system.plantName] + [x + '_' + h for x in
                                                                                               ['Spot_Exp_USH',
                                                                                                'Spot_Exp_ABH',
                                                                                                'load_x', 'POTHH_SH']],
                                      [1.] * len(self.system.plantName) + [-1., -1., 1., 1.],
                                      'E',
                                      self.load_resource_balance_rhs[i] ))

        return constraints


    def get_var_bounds(self):
        var_bounds = dict()
        var_bounds['Q_TY_ARD_now'] = 'free'
        var_bounds['V_TY_total'] = 'free'
        var_bounds['V_TY_total_sv'] = 'free'
        var_bounds['V_TY_sys'] = 'free'
        var_bounds['V_TY_MCA_now'] = 'free'
        var_bounds['V_TY_MCA_sv'] = 'free'
        for h in self.hplName:
            var_bounds['Q_TY_MCA_now_' + h] = 'free'
        var_bounds['V_NT_BC_sv'] = 'free'
        var_bounds['Q_NT_BC_now'] = 'free'
        var_bounds['Q_NT_US_now'] = 'free'
        for plant in self.fccPlantName:
            var_bounds['fcc_1_' + plant] = 'free'
        var_bounds['MC_inflow'] = 'free'
        var_bounds['SR_inflow'] = 'free'
        var_bounds['TD_inflow'] = 'free'
        var_bounds['MC_se_fc_Jul'] = 'free'
        var_bounds['CR_se_fc_Jul'] = 'free'
        var_bounds['TD_se_fc_Jul'] = 'free'
        var_bounds['SR_se_fc_Jul'] = 'free'
        var_bounds['sf_TD'] = 'free'
        var_bounds['sa_TD'] = 'free'
        return var_bounds

    def loadParameters(self,path, fccPlantName):
        # reading data
        dataRaw = readFileOfTables( path )

        sf_TD = indexing(dataRaw['sf_TD'],MONTH_LIST)
        self.params.sf_TD_intercept = sf_TD[:,0]
        self.params.sf_TD_slope = sf_TD[:,1]

        sa_TD = indexing(dataRaw['sa_TD'],MONTH_LIST)
        self.params.sa_TD_intercept = sa_TD[:,0]
        self.params.sa_TD_slope = sa_TD[:,1]

        self.params.fcc_intercept = indexing(dataRaw['fcc_intercept_month'], MONTH_LIST)
        self.params.fcc_slope = indexing(dataRaw['fcc_slope_month'], MONTH_LIST)
        self.params.fcc_max = indexing(dataRaw['fcc_max_month'], MONTH_LIST)
        self.params.fcc_min = indexing(dataRaw['fcc_min_month'], MONTH_LIST)

        X0_1 = indexing(dataRaw['X0_1'],MONTH_LIST)
        self.params.X0_1 = X0_1[:,0]
        self.params.X1_1 = X0_1[:,1]
        self.params.X2_1 = X0_1[:,2]
        self.params.X3_1 = X0_1[:,3]
        self.params.X4_1 = X0_1[:,4]
        self.params.X5_1 = X0_1[:,5]

        X0_2 = indexing(dataRaw['X0_2'],MONTH_LIST)
        self.params.X0_2 = X0_2[:,0]
        self.params.X1_2 = X0_2[:,1]
        self.params.X2_2 = X0_2[:,2]
        self.params.X3_2 = X0_2[:,3]
        self.params.X4_2 = X0_2[:,4]
        self.params.X5_2 = X0_2[:,5]

        self.params.D1 = indexing(dataRaw['D1_month'], MONTH_LIST, default = 9999999)
        self.params.D2 = np.ones([len(MONTH_LIST)]) * 9999999
        self.params.D3 = indexing(dataRaw['D3_month'], MONTH_LIST, default = 9999999)
        self.params.D4 = indexing(dataRaw['D4_month'], MONTH_LIST, default = 9999999)
        self.params.D5 = np.ones([len(MONTH_LIST)]) * 9999999

        MC_inflow = indexing(dataRaw['MC_inflow'],MONTH_LIST)
        self.params.MC_inflow_intercept = MC_inflow[:,0]
        self.params.MC_inflow_slope = MC_inflow[:,1]

        SR_inflow = indexing(dataRaw['SR_inflow'],MONTH_LIST)
        self.params.SR_inflow_intercept = SR_inflow[:,0]
        self.params.SR_inflow_slope = SR_inflow[:,1]

        TD_inflow = indexing(dataRaw['TD_inflow'],MONTH_LIST)
        self.params.TD_inflow_intercept = TD_inflow[:,0]
        self.params.TD_inflow_slope = TD_inflow[:,1]

        CR_se_fc_Jul = indexing(dataRaw['CR_se_fc_Jul'],MONTH_LIST)
        self.params.CR_se_fc_Jul_intercept = CR_se_fc_Jul[:,0]
        self.params.CR_se_fc_Jul_slope = CR_se_fc_Jul[:,1]

        MC_se_fc_Jul = indexing(dataRaw['MC_se_fc_Jul'],MONTH_LIST)
        self.params.MC_se_fc_Jul_intercept = MC_se_fc_Jul[:,0]
        self.params.MC_se_fc_Jul_slope = MC_se_fc_Jul[:,1]

        SR_se_fc_Jul = indexing(dataRaw['SR_se_fc_Jul'],MONTH_LIST)
        self.params.SR_se_fc_Jul_intercept = SR_se_fc_Jul[:,0]
        self.params.SR_se_fc_Jul_slope = SR_se_fc_Jul[:,1]

        TD_se_fc_Jul = indexing(dataRaw['TD_se_fc_Jul'],MONTH_LIST)
        self.params.TD_se_fc_Jul_intercept = TD_se_fc_Jul[:,0]
        self.params.TD_se_fc_Jul_slope = TD_se_fc_Jul[:,1]

        V_SOA_LCA = indexing(dataRaw['V_SOA_LCA'],MONTH_LIST)
        self.params.V_SOA = V_SOA_LCA[:,0]
        self.params.V_LCA = V_SOA_LCA[:,1]

        V_TY = indexing(dataRaw['V_TY'],MONTH_LIST)
        self.params.V_TY_total_max = V_TY[:,0]
        self.params.V_TY_total_min = V_TY[:,1]

        Q_NT_BC = indexing(dataRaw['Q_NT_BC'],MONTH_LIST )
        self.params.Q_NT_BC_min = Q_NT_BC[:,0]
        self.params.Q_NT_BC_max = Q_NT_BC[:,1]

        self.params.V_NT_BC_min = np.ones(len(MONTH_LIST)) * 10707.31 
        self.params.V_NT_BC_max = np.ones(len(MONTH_LIST)) * 32121.92

        HKARD_MidC = indexing(dataRaw['HKARD_MidC'],MONTH_LIST)
        self.params.HKARD = HKARD_MidC[:,0]
        self.params.price_Mid_C = HKARD_MidC[:,1]

        self.params.NT_LOSS = np.zeros(len(MONTH_LIST))

        self.params.V_TY_max = np.zeros([len(MONTH_LIST) , len(self.fccPlantName)] )
        for t in range(len(MONTH_LIST)):
            self.params.V_TY_max[t] = [ self.V_TY_MCA_max - max(0,self.params.V_SOA[t]) , self.V_TY_ARD_max - max(0,self.params.V_LCA[t]) ]
        
        self.params.Q_SOA = np.zeros(len(MONTH_LIST))
        for t in range(len(MONTH_LIST)):
            self.params.Q_SOA[t] = ( self.params.V_SOA[t-1] - self.params.V_SOA[t] ) / dayPerMonth[t]

        self.params.Q_LCA = np.zeros(len(MONTH_LIST))
        for t in range(len(MONTH_LIST)):
            self.params.Q_LCA[t] = ( self.params.V_LCA[t-1] - self.params.V_LCA[t] ) / dayPerMonth[t]

        self.params.V_treaty_select = indexing(dataRaw['V_treaty_select'],MONTH_LIST)

        self.params.V_SOA_ini = 0
        self.params.V_LCA_ini = -2245

        for t in range(len(MONTH_LIST)):
            self.params.storage_default[t][self.params.plant_storage.index('MCA')] = self.params.V_treaty_select[t][0] + 2 * self.params.V_treaty_select[t][2] + self.params.V_SOA[t] + self.params.V_MCA_recall + self.params.VMin1[self.system.plantName.index('MCA')]
            self.params.storage_default[t][self.params.plant_storage.index('ARD')] = self.params.V_treaty_select[t][1] - self.params.V_treaty_select[t][0] + self.params.V_LCA[t] + self.params.VMin1[self.system.plantName.index('ARD')]

        self.params.treaty_multi_ini = np.array(dataRaw['treaty_multi_ini'].value, dtype=float)