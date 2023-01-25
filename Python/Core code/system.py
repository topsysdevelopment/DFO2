from readFileTest import *
from model import Module, State, Variable
from solver import linear_constraint, concave_piecewise_linear_contraint
import copy

MONTH_LIST = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
dayPerMonth = [31,28,31,30,31,30,31,31,30,31,30,31]
        
class System(Module):

    def __init__(self, dataPath , inflowModel ):
        super(System, self).__init__()
        self.type = 'system'

        # CRTOM
        self.plantName = ['GMS', 'PCN' , 'STC' , 'MCA', 'REV' , 'ARD']
        self.params.plantName = self.plantName
        self.fccPlantName = ['MCA', 'ARD']
        self.params.fccPlantName = self.fccPlantName
        self.params.hplName = ['PEAK' , 'HLH1' , 'HLH2' , 'LLH1' , 'LLH2' ]
        self.startMonth = 'Dec' #
        self.params.T = 1*12 # optimization horizon #TODO outside model
        self.buildTimeStepProperty()
        # Case Study
        self.plant_storage = ['GMS', 'MCA', 'ARD' ]
        self.params.plant_storage = ['GMS', 'MCA', 'ARD']
        self.id_plant_storage = [self.plantName.index(plant) for plant in self.plant_storage]
        self.discountRate = 0.07 # TODO : parameter
        self.params.discount = 1/ (1 + self.discountRate/ len(MONTH_LIST) ) # firest order approx

        # load Data
        self.loadData( dataPath )

        # state value and variables
        self.storage = np.zeros(len(self.plantName))

        state_variable = Variable([plant + '_storage_sv' for plant in self.plant_storage])
        #initialStorage = self.params.V00[self.id_plant_storage]
        self.initialStorage = np.array([477712., 211909.,36823. ])
        self.module_state = State(state_variable, self.initialStorage, self.params.storage_default)

    def get_initial_state(self, mode):
        if mode == 'unique':
            ini_storage = copy.copy(self.initialStorage)
            return ini_storage
        elif mode == 'multi_random':
            return [ np.random.choice(self.params.storage_GMS_ini_multi[:,0]), 0., 0.]
        else:
            print("initialization mode not managed") 

    def update_initial_state(self, state_vec):
        # treaty_id = self.params.type_vec.index('treaty')
        # system_id = self.params.type_vec.index('system')
        # treaty_state = state_vec[treaty_id]
        # # update MCA and ARD strorage select:
        # state_vec[system_id][self.params.plant_storage.index('MCA')] \
        #    = (treaty_state[1] + 2 * treaty_state[2] + self.params.V_SOA_ini + self.params.V_MCA_recall \
        #      + self.params.VMin1[self.plantName.index('MCA')])[0]

        # state_vec[system_id][self.params.plant_storage.index('ARD')] \
        #     = (treaty_state[0] - treaty_state[1] + self.params.V_LCA_ini + self.params.VMin1[
        #     self.plantName.index('ARD')])[0]
        return state_vec

    def map_state_module_value(self):
        self.storage[self.id_plant_storage] = self.module_state.value
        self.global_state.storage = self.storage

    def updateValues(self):
        t_now = self.global_state.currentMonth
        #self.storage_no_fcc_rhs[:] = [ 'Q_reservoir_' + plantName] * (self.params.timeStepProp_hours[t_now] / 24)


    def get_constraints(self):# , current_month):

        t_now = self.global_state.currentMonth
        self.updateValues()
        constraints = []
        ######################
        # static constraints #
        ######################
        # Storage lower limit

        # TODO : MCA and ARD no sv constraints, only GMS
        for i,plant in enumerate(self.plant_storage):
        	if plant not in self.fccPlantName:
	            id_plant = self.plantName.index(plant)
	            constraints.append(
	                linear_constraint( [ plant + '_storage_sv'] , [1.] ,'E', self.storage[id_plant:id_plant+1], state_variable =  plant + '_storage_sv'  )  )
        for plant in self.plant_storage:
            constraints.append( 
                linear_constraint( [ plant + '_storage_next_ts'] , [1] ,'G', self.params.VMin1[self.plantName.index(plant)] ) ) 
        # storage upper limit for plants no FCC
        for plant in self.plant_storage:
            if plant not in self.fccPlantName:
                constraints.append( 
                    linear_constraint( [ plant + '_storage_next_ts'] , [1] , 'L', self.params.VMax1[self.plantName.index(plant)] ) )
                #constraints.append(
                #    linear_constraint([plant + '_storage_next_ts'], [1], 'L', self.params.VMax1[self.plantName.index(plant)]))
        # turbine lower and upper limit
        for plant in self.plantName:
            for i,h in enumerate(self.params.hplName):
                constraints.append( 
                    linear_constraint( [ 'QT_' + plant +'_'+ h ] , [1] ,'G',self.params.QTMin[self.plantName.index(plant),i] ) )
                constraints.append( 
                    linear_constraint( [ 'QT_' + plant +'_'+ h , 'QT_max_'+ plant ], [1 , -1] ,'L', np.array([0.]) ) )
        for plant in self.plantName:
            if plant in self.plant_storage:
                constraints.append( 
                    concave_piecewise_linear_contraint( varx = plant + '_storage_average'    , 
                                            vary = 'QT_max_' + plant ,
                                            nparts = self.params.QT_max_piece_n[self.plant_storage.index(plant)], 
                                            bkpointx = self.params.QT_max_bkpt[t_now,self.plant_storage.index(plant)],
                                            slope = self.params.QT_max_slope[t_now,self.plant_storage.index(plant)] * self.params.P_maintenance[t_now,self.plantName.index(plant)],#,self.plantName.index(plant)] , 
                                            intercept = self.params.QT_max_intercept[self.plant_storage.index(plant)] * self.params.P_maintenance[t_now,self.plantName.index(plant)]) )#,self.plantName.index(plant)] )
            else:
                constraints.append( 
                    linear_constraint(['QT_max_'+ plant ], [1],'L', self.params.QT_max[self.plantName.index(plant)] * self.params.P_maintenance[t_now,self.plantName.index(plant)] ))

        # Spill limit
            #lower limit
        for plant in self.plantName:
            constraints.append(
                linear_constraint( [ 'QS_' + plant , 'QS_min_x_' + plant , 'QS_max_x_' + plant  ], [1,1 , -1] , 'G', self.params.QSMIN[t_now,self.plantName.index(plant)] ) )
            constraints.append(
                linear_constraint( [ 'QS_' + plant , 'QS_min_x_' + plant , 'QS_max_x_' + plant , 'QS_max_' + plant ] , [1 ,1 , -1, -1 ] , 'L', np.array([0.]) ) )

            # upper limit
        for plant in self.plantName:
            if plant in self.plant_storage:
                constraints.append( 
                    linear_constraint(  [ 'QS_max_' + plant , plant + '_storage_average' ] , [1 , - self.params.QS_max_slope[t_now,self.plant_storage.index(plant)] ] ,'E', self.params.QS_max_intercept[t_now,self.plantName.index(plant)] ) )
            else:
                constraints.append( 
                    linear_constraint(  [ 'QS_max_' + plant ] , [1 ] ,'E', self.params.QS_max[self.plantName.index(plant)] ) )

        # Plant release limits, penalty variables for relaxtion
        for plant in self.plantName:
            for i,h in enumerate(self.params.hplName):
                constraints.append( 
                    linear_constraint( [ 'QT_' + plant +'_'+ h ,  'QS_' + plant , 'QP_' + plant +'_'+ h ] , [1 , 1, -1 ] , 'E', np.array([0.]) ) )
                constraints.append(
                    linear_constraint( [ 'QP_' + plant +'_'+ h , 'QP_min_x_' + plant + '_' + h , 'QP_max_x_' + plant + '_' + h  ] , [1,1,-1] , 'L', self.params.QP_MAX[t_now,self.plantName.index(plant)] ) )
                constraints.append(
                    linear_constraint( [ 'QP_' + plant +'_'+ h , 'QP_min_x_' + plant + '_' + h , 'QP_max_x_' + plant + '_' + h  ] , [1,1,-1] , 'G', self.params.QP_MIN[t_now,self.plantName.index(plant)] ) )
        # Generation 
        for plant in self.plantName:
            for i,h in enumerate(self.params.hplName):
                #constraints.append( 
                #    linear_constraint(['production_' + plant + '_' + h, 'QT_' + plant +'_'+ h ] , [ 1, - self.params.HK[t_now,self.plantName.index(plant)] ] , 'E', np.array([0.]) ) )
                constraints.append( 
                    linear_constraint(['production_' + plant + '_' + h], [1] , 'G', self.params.P_min[self.plantName.index(plant)] ) )
                constraints.append(
                    linear_constraint(['production_' + plant + '_' + h, 'production_max_' + plant] , [ 1, - self.params.P_availability[self.plantName.index(plant),0] ] , 'L', np.array([0.]) ) )
            if plant in self.plant_storage:
                constraints.append(
                    concave_piecewise_linear_contraint( varx = plant + '_storage_average',
                                            vary = 'production_max_' + plant,
                                            nparts = self.params.P_max_piece_n[self.plant_storage.index(plant)],
                                            bkpointx = self.params.P_max_bkpt[self.plant_storage.index(plant)],
                                            slope = self.params.P_max_slope[self.plant_storage.index(plant)] * self.params.P_maintenance[t_now,self.plantName.index(plant)],
                                            intercept = self.params.P_max_intercept[self.plant_storage.index(plant)] * self.params.P_maintenance[t_now,self.plantName.index(plant)] ) )
            else:
                # TODO : move into bounds
                constraints.append(
                    linear_constraint( [ 'production_max_' + plant], [1], 'L', self.params.P_max[self.plantName.index(plant)] * self.params.P_maintenance[t_now,self.plantName.index(plant)] ) )


        # TODO : move into bounds
        for i,h in enumerate(self.params.hplName):
            constraints.append( 
                linear_constraint( ['Spot_Exp_USH_' + h ],[1], 'G' , self.params.US_tran_minH[t_now,i] ) )
            constraints.append( 
                linear_constraint( ['Spot_Exp_USH_' + h ],[1], 'L' , self.params.US_tran_maxH[t_now,i] ) )
            constraints.append( 
                linear_constraint( ['Spot_Exp_ABH_' + h ],[1], 'G' , self.params.AB_tran_minH[t_now,i] ) )
            constraints.append( 
                linear_constraint( ['Spot_Exp_ABH_' + h ],[1], 'L' , self.params.AB_tran_maxH[t_now,i] ) )

        for i, plant in enumerate(self.plantName):
            if plant not in self.fccPlantName:
                var, coeff = self.returnMassBalanceVariableAndCoeff(plant, t_now)
                if plant in self.plant_storage:
                    constraints.append( 
                        linear_constraint( var + [ plant + '_storage_next_ts' , plant + '_storage_sv',  'Q_reservoir_' + plant] , coeff  + [ 1. , -1., - self.params.timeStepProp_hours[t_now] / 24 ], 'E', 0. ) )
                else:
                    constraints.append( 
                        linear_constraint( var + [ 'Q_reservoir_' + plant] , coeff + [- self.params.timeStepProp_hours[t_now] / 24 ] , 'E', 0. ) )

        # average storage
        for plant in self.plant_storage:
            if plant not in self.fccPlantName:
                constraints.append( 
                    linear_constraint( [plant + '_storage_average' , plant + '_storage_next_ts' , plant + '_storage_sv'] , [1 , -1./2 , -1./2] , 'E' , 0. ) )

        
        return constraints

    def get_var_bounds(self):
        variable_bounds = dict()
        for plant in self.plantName:
            for i,h in enumerate(self.params.hplName):
                variable_bounds['QP_' + plant +'_'+ h ] = 'free'
            variable_bounds['QS_' + plant ] = 'free'
            variable_bounds['QT_max_'+ plant ] = 'free'
            variable_bounds['QS_max_'+ plant ] = 'free'
        for plant in self.plant_storage:
            variable_bounds[ plant + '_storage_next_ts'] = 'free'
            variable_bounds[ plant + '_storage_average'] = 'free'
            variable_bounds[ plant + '_storage_sv'] = 'free'
        for h in self.params.hplName:
            variable_bounds['POTHH_SH_' + h ] = 'free'

        return variable_bounds

    def returnMassBalanceVariableAndCoeff(self, plant , current_month):
        t_now = current_month #self.global_state.currentMonth
        var = []
        coeff = []
        for i in range(len(self.params.flow_matrix)):
            # check inflow in flow matrix
            if plant == self.params.flow_matrix[i][0]:
                var.append( 'QS_' + plant )
                coeff.append( ( - self.params.flow_matrix[i][2] * self.params.timeStepProp_hours[t_now] ) * (-1./24) )
                for j,h in enumerate(self.params.hplName):
                    var.append( 'QT_' + plant +'_'+ h )
                    coeff.append( ( -  self.params.flow_matrix[i][2] * self.params.HPLHR[t_now,j]  ) * (-1./24) )
            # check outflows in flow matrix
            elif plant == self.params.flow_matrix[i][1]:
                upStreamPlant = self.params.flow_matrix[i][0]
                var.append( 'QS_' + upStreamPlant )
                coeff.append( (  self.params.flow_matrix[i][2] * self.params.timeStepProp_hours[t_now] ) * (-1./24) )
                for j,h in enumerate(self.params.hplName):
                    var.append( 'QT_' + upStreamPlant +'_'+ h )
                    coeff.append( (   self.params.flow_matrix[i][2] * self.params.HPLHR[t_now,j]  ) * (-1./24) )

        return var, coeff

    # apply transition to the system
    def applyTransition(self, decision, dict_variable_list):
        for i,plant in enumerate(self.plant_storage):
            var_next_ts = self.module_state.state_variable.name_next_ts[i]
            self.storage[self.id_plant_storage[i]] = decision[dict_variable_list[var_next_ts]]
            self.module_state.value[i] = decision[dict_variable_list[var_next_ts]]
        self.global_state.storage = self.storage
        
    #def set_state_value(self,state_value):
    #    self.module_state.value = state_value
    #    self.storage[self.id_plant_storage] = state_value

    # TODO : move to global
    def buildTimeStepProperty(self):
        startYearInflowModel = 'Oct'
        idFirstMonth = MONTH_LIST.index(self.startMonth)
        self.params.timeStepProp_hours = [24*x  for x in dayPerMonth ] # [24* dayPerMonth[(idFirstMonth+x)%12] for x in range(self.params.T) ]
        self.params.timeStepProp_year = [ 2 + math.floor((x-MONTH_LIST.index(startYearInflowModel)-1)/12) for x in range(self.params.T)] # first year is 1
        self.params.timeStepProp_month = [ (x+idFirstMonth)%12 for x in range(self.params.T) ] # january index is 0

    #
    def loadData(self, path):
        # reading data
        dataRaw = readFileOfTables( path )
        # indexing data
        self.params.availability = indexing(dataRaw['Availability_month'],MONTH_LIST)
        self.params.P_maintenance = self.params.availability
        self.params.HPLHR = indexing(dataRaw['HPLHR_month'],MONTH_LIST)
        self.params.HK = indexing(dataRaw['HK_month'],MONTH_LIST)
        self.params.LOADH = indexing(dataRaw['LOADH_month'],MONTH_LIST)
        self.params.POTHH_SH_intercept = indexing(dataRaw['POTHH_SH_intercept_month'],MONTH_LIST)
        self.params.POTHH_SH_slope = indexing(dataRaw['POTHH_SH_slope_month'],MONTH_LIST)
        self.params.POTHH_non_SH = indexing(dataRaw['POTHH_non_SH_month'],MONTH_LIST)
        self.params.price_Exp_USH = indexing(dataRaw['price_Exp_USH_month'],MONTH_LIST)
        self.params.price_Imp_USH = indexing(dataRaw['price_Imp_USH_month'],MONTH_LIST)
        self.params.US_tran_minH = indexing(dataRaw['US_tran_minH_month'],MONTH_LIST)
        self.params.US_tran_maxH = indexing(dataRaw['US_tran_maxH_month'],MONTH_LIST)
        self.params.QT_max_intercept = indexing(dataRaw['QT_max_intercept'], self.plant_storage )
        self.params.QT_max_piece_n = indexing(dataRaw['QT_max_piece_n'],self.plant_storage)
        self.params.QT_max_bkpt = indexing(dataRaw['QT_max_bkpt_month'],MONTH_LIST, self.plant_storage)
        self.params.QT_max_slope = indexing(dataRaw['QT_max_slope_month'],MONTH_LIST, self.plant_storage )
        self.params.QS_max_intercept = indexing(dataRaw['QS_max_intercept_month'],MONTH_LIST)
        self.params.QS_max_slope = indexing(dataRaw['QS_max_slope_month'],MONTH_LIST)
        self.params.P_max = indexing(dataRaw['P_max'],self.plantName)
        self.params.P_availability = indexing(dataRaw['P_availability'],self.plantName)
        self.params.P_max_intercept = indexing(dataRaw['P_max_intercept'],self.plant_storage)
        self.params.P_max_piece_n = indexing(dataRaw['P_max_piece_n'],self.plant_storage)
        self.params.P_max_bkpt = indexing(dataRaw['P_max_bkpt'],self.plant_storage)
        self.params.P_max_slope = indexing(dataRaw['P_max_slope'],self.plant_storage)
        self.params.QT_max = indexing(dataRaw['QT_max'],self.plantName)
        #self.params.QT_max_piece_n = indexing(dataRaw['QT_max_piece_n'],self.plant_storage)
        self.params.QS_max = indexing(dataRaw['QS_max'],self.plantName)
        self.params.flow_matrix = [['GMS','PCN',1],['PCN','STC',1],['STC','border',1],['MCA','REV',1],['REV','ARD',1],['ARD','border',1]]
        self.params.V00 = np.array(dataRaw['V00'].value,dtype='float')
        self.params.VMax1 = np.array(dataRaw['VMax1'].value,dtype='float')
        self.params.VMin1 = np.array(dataRaw['VMin1'].value,dtype='float')
        self.params.V_Min = np.array(dataRaw['V_Min'].value,dtype='float')
        self.params.price_Exp_ABH = self.params.price_Exp_USH + 4.33
        self.params.price_Imp_ABH = self.params.price_Exp_ABH + 2.52 # TODO imp or exp ?
        self.params.QTMin = np.zeros([len(self.plantName), len(self.params.hplName)] )
        self.params.QSMIN = np.zeros([len(MONTH_LIST),len(self.plantName)] )
        self.params.QP_MAX = np.ones([len(MONTH_LIST),len(self.plantName)]) * 50000.
        self.params.QP_MIN = indexing(dataRaw['QP_Min_month'],MONTH_LIST)
        self.params.P_min = np.zeros([len(self.plantName)])
        self.params.P_Imports = np.zeros([len(MONTH_LIST), len(self.params.hplName)] )
        self.params.P_Exports = np.zeros([len(MONTH_LIST), len(self.params.hplName)] )
        self.params.AB_tran_minH = np.zeros([len(MONTH_LIST), len(self.params.hplName)] )
        self.params.AB_tran_maxH = np.zeros([len(MONTH_LIST), len(self.params.hplName)] )
        self.params.price_multiplier =  np.ones(len(self.params.hplName))
        self.params.storage_default = indexing(dataRaw['storage_default'],MONTH_LIST )
        self.params.storage_GMS_ini_multi = np.array(dataRaw['storage_ini_multi']['GMS_ini_multi'].value, dtype='float')