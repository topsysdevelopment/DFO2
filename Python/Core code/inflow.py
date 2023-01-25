import numpy as np
import random 
import math
from functools import reduce
from model import Module, State
from operator import mul
from solver import linear_constraint

MONTH_LIST = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

##############################################
### Inflow Model Class ####
##############################################
class Inflow(Module):

    def __init__(self, path):
        super(Inflow,self).__init__()
        self.type = 'inflow_model'

        self.path = path
        self.parameters = inflowModelParameters(path)
        self.yearPlanStart = 2017 # TODO : move in parameters.py
        self.yearDataStart = 1929 # TODO : move in parameters.py
        #self.currentYear = self.yearDataStart

        self.dim_random_process = 4 #right ?

        self.se_fc_rwn = np.zeros(len(self.parameters.riverName))
        self.se_fc_resi = np.zeros(len(self.parameters.riverName))
        self.se_actual = np.zeros(len(self.parameters.riverName))
        self.se_fc_reservoir = np.zeros(len(self.parameters.riverName))

        self.resi = np.zeros(len(self.parameters.riverName))
        self.resi_sv = np.zeros(len(self.parameters.riverName))
        self.inflow_corrected = np.zeros(len(self.parameters.riverName))
        # TODO : split inflow and forecast in two class heriting from class random process
        self.inflow= np.zeros( len(self.parameters.plantName) )
        self.forecast = np.zeros( len(self.parameters.riverName) )
        self.inflowRiver = np.zeros(len(self.parameters.riverName))
        self.se_fc = np.zeros(len(self.parameters.riverName))
        self.indexedSample = False
        self.sampleIndex = np.zeros(4)
        self.inflowResidual = []

        # declare random variables
        self.random_variable.riverInflows = self.inflowRiver
        self.random_variable.inflows = self.inflow
        self.random_variable.forecast = self.forecast
        self.random_variable.se_fc = self.se_fc
        # TODO change
        self.forecastResidual = np.zeros( len(self.parameters.riverName) )
        self.previousForecastResidual = np.zeros( len(self.parameters.riverName) )
        self.intialForecastResidual = np.zeros( len(self.parameters.riverName) )

        self.inflowResidual = np.zeros( len(self.parameters.riverName) )
        self.previousInflowResidual = np.zeros( len(self.parameters.riverName) )
        self.initialInflowResidual = np.zeros( len(self.parameters.riverName) )

        self.inflowRandomSampleValue = np.zeros( 2 ) # last sample value
        self.inflowRandomSampleProbability = 1 # last sample probability

        self.forecastRandomSampleValue = np.zeros(2) # last sample value
        self.forecastRandomSampleProbability = 1 

        # TODO : method in GlobalModel / Class random process
        self.month_stochastic_forecast = self.parameters.se_fc_rwn_month
        self.month_stochastic_inflows = self.parameters.stage_rand_inflow
        #self.numberPossibleSample = np.ones(len(MONTH_LIST), dtype=np.int)
        #for i,month in enumerate(MONTH_LIST):
        #    if month in self.month_stochastic_forecast:
        #        self.numberPossibleSample[i] *= 3**2
        #    if month in self.month_stochastic_inflows:
        #        self.numberPossibleSample[i] *= 3**2

        self.number_possible_sample_vec = [[1]*4 for _ in MONTH_LIST]
        for i, month in enumerate(MONTH_LIST):
            if month in self.month_stochastic_forecast:
                self.number_possible_sample_vec[i][0:2] = [3,3]
            if month in self.month_stochastic_inflows:
                self.number_possible_sample_vec[i][2:4] = [3,3]

        self.numberPossibleSample = [reduce(mul,self.number_possible_sample_vec[i]) for i in range(len(MONTH_LIST))]

        # TODO : declare only size of var ?
        # declare state
        default_state = [np.zeros(2),np.zeros(2)]*12
        self.module_state = State( [], [self.intialForecastResidual , self.initialInflowResidual], default_state)

    def get_initial_state(self, mode):
        return [self.intialForecastResidual , self.initialInflowResidual]

    def map_state_module_value(self):
        self.forecastResidual[:], self.inflowResidual[:] = self.module_state.value

#    def initialize(self):
#        self.forecastResidual, self.inflowResidual = self.module_state.ini_value
#
    def set_state_value(self,state_value):
        self.module_state.value = state_value
        self.previousForecastResidual[:], self.previousInflowResidual[:] = state_value

    def get_constraints(self):
        t_now = self.global_state.currentMonth
        constraints = []

        month_name = MONTH_LIST[self.global_state.currentMonth]
        self.forecastRandomSampleProbability = 1

    ### Seasonal inflow forecast of GMS and MCA
        for i, river in enumerate(self.parameters.riverName):
            l = linear_constraint( [ "previous_forecast_residual_sv_" + river ] , [1.] ,'E', self.previousForecastResidual[i], state_variable =  "previous_forecast_residual_sv_" + river   )
            constraints.append(l)

        for i, river in enumerate(self.parameters.riverName):
            if month_name in self.parameters.se_fc_month:
                if month_name in self.parameters.se_fc_gen_month:
                    if month_name == self.parameters.se_fc_start_month:
                        l = linear_constraint( [ "se_fc_resi_" + river ] , [1.] ,'E', self.se_fc_rwn[i]  )
                    else:
                        if month_name in self.parameters.se_fc_rwn_month :
                            l = linear_constraint( [ "se_fc_resi_" + river,  "previous_forecast_residual_sv_" + river ] , [1., - self.parameters.forecastRou[self.global_state.currentMonth][i] ] ,'E', np.sqrt( 1 - self.parameters.forecastRou[self.global_state.currentMonth][i] **2 ) * self.se_fc_rwn[i]  )
                        else:
                            l = linear_constraint( [ "se_fc_resi_" + river,  "previous_forecast_residual_sv_" + river ] , [1., - self.parameters.forecastRou[self.global_state.currentMonth][i] ] ,'E', 0. )
                else : 
                    l = linear_constraint( [ "se_fc_resi_" + river,  "previous_forecast_residual_sv_" + river ] , [1., -1. ] ,'E', 0.  )
                constraints.append(l)


        for i, river in enumerate(self.parameters.riverName):
            if month_name in self.parameters.se_fc_month:
                l = linear_constraint( [ "se_fc_reservoir_" + river, "se_fc_resi_" + river] , [ 1., - self.parameters.forecastReservoirStd[self.global_state.currentMonth][i] ] ,'E', self.parameters.forecastReservoirIntercept[self.global_state.currentMonth][i] + self.parameters.forecastReservoirSlope[self.global_state.currentMonth][i] * (self.yearPlanStart + self.global_state.currentYear -1) )
            else:
                l = linear_constraint( [ "se_fc_resi_" + river] , [ 1.], 'E', 0. )
            constraints.append(l)



    ### Seasonal inflow forecast of Peace and Columbia
        for i, river in enumerate(self.parameters.riverName):
            if month_name in self.parameters.se_fc_month:
                l = linear_constraint( [ "se_fc_" + river ,  "se_fc_reservoir_" + river ] , [1. , - self.parameters.forecastSlope[self.global_state.currentMonth][i] ] ,'E', self.parameters.forecastIntercept[self.global_state.currentMonth][i]  )
                constraints.append(l)

    ### Seasonal inflow 
        for i, river in enumerate(self.parameters.riverName):
            if month_name in self.parameters.se_fc_month:
                l = linear_constraint( [ "se_actual_" + river, "se_fc_" + river ] , [1., - self.parameters.forecastActualSlope[self.global_state.currentMonth][i] ] ,'E', self.parameters.forecastActualIntercept[self.global_state.currentMonth][i]   )
            else:
                l = linear_constraint( [ "se_actual_" + river ] , [1.] ,'E', 0.   )
            constraints.append(l)


#####
        
        previous_month_name = MONTH_LIST[ self.global_state.currentMonth - 1 ]
        
        for i, river in enumerate(self.parameters.riverName):
            if month_name in self.parameters.stage_rand_inflow and previous_month_name in self.parameters.stage_rand_inflow :
                l = linear_constraint( [ "resi_sv_" + river ] , [1.] ,'E', self.previousInflowResidual[i], state_variable =  "resi_sv_" + river  )
            else:
                l = linear_constraint( [ "resi_sv_" + river ] , [1.] ,'E', 0., state_variable =  "resi_sv_" + river   )
            constraints.append(l)

        for i, river in enumerate(self.parameters.riverName):
            if month_name in self.parameters.stage_rand_inflow:
                if previous_month_name not in self.parameters.stage_rand_inflow:
                    l = linear_constraint( [ "resi_" + river ] , [1.] ,'E', self.rwn[i]  )
                else:
                    l = linear_constraint( [ "resi_" + river, "resi_sv_" + river ] , [1., - self.parameters.inflowRou[self.global_state.currentMonth][i] ] ,'E', np.sqrt(1-self.parameters.inflowRou[self.global_state.currentMonth][i]**2) * self.rwn[i] )
            else:
                l = linear_constraint( [ "resi_" + river ] , [1.] ,'E', 0.  )
            constraints.append(l)

        inflow_min = 0.
        for i, river in enumerate(self.parameters.riverName):
            if self.inflow_corrected[i] == 0.:
                var = ['Q_river_' + river ]
                coeff = [1.]
                rhs = self.parameters.inflowIntercept[self.global_state.currentMonth][i] + self.parameters.inflowSlopeYear[self.global_state.currentMonth][i] * (self.yearPlanStart + self.global_state.currentYear -1 )
                if month_name in self.parameters.stage_inflow_se:
                    var.append( "se_actual_" + river )
                    coeff.append( - self.parameters.inflowSlopeSeasonal[self.global_state.currentMonth][i] )
                if month_name in self.parameters.stage_rand_inflow:
                    var.append( "resi_" + river )
                    coeff.append( - self.parameters.inflowStd[self.global_state.currentMonth][i] )
            else:
                var = ['Q_river_' + river ]
                coeff = [1.]
                rhs = inflow_min
            constraints.append(linear_constraint( var , coeff ,'E', rhs ))        


        for plantIndex, plantName in enumerate(self.parameters.plantName):
            if plantName in self.parameters.plantInflowVaryName :
                indexInflowPlantVary = self.parameters.plantInflowVaryName.index(plantName)
                for riverIndex,plantsForEachRiver in enumerate(self.parameters.plantRiver):
                    if plantName in plantsForEachRiver:
                        l = linear_constraint( [ 'Q_reservoir_' + plantName , 'Q_river_' + self.parameters.riverName[riverIndex] ] , [1., -self.parameters.inflowPercentage[self.global_state.currentMonth,indexInflowPlantVary]] ,'E', 0. )
                        break
            else:
                indexFixedInflow = self.parameters.plantInflowFixedName.index(plantName)
                l = linear_constraint( [ 'Q_reservoir_' + plantName] , [1.] ,'E', self.parameters.plantInflowFixedValue[indexFixedInflow] )
            constraints.append( l )

        # Calculate small hydro from actual seasonal inflow    
        for i,h in enumerate(self.params.hplName):
            constraints.append( 
                    linear_constraint( ['POTHH_SH_' + h ] + [ "se_actual_" + river for river in self.parameters.riverName ],[1, - self.params.POTHH_SH_slope[t_now,i], - self.params.POTHH_SH_slope[t_now,i] ], 
                                            'E' , self.params.POTHH_SH_intercept[t_now,i]  )  )
        
        return constraints

    def get_var_bounds(self):
        variable_bounds = dict()
        for river in self.parameters.riverName:
            variable_bounds[ "previous_forecast_residual_sv_" + river ] = 'free'
            variable_bounds[ "se_fc_resi_" + river ] = 'free'
            variable_bounds[ "se_fc_reservoir_" + river] = 'free'
            variable_bounds[ "se_fc_" + river ] = 'free'
            variable_bounds[ "se_actual_" + river ] = 'free'
            variable_bounds[ "resi_sv_" + river ] = 'free'
            variable_bounds[ "resi_" + river ] = 'free'
            variable_bounds[ 'Q_river_' + river ] = 'free'
            for plantName in self.parameters.plantName:
                variable_bounds[ 'Q_reservoir_' + plantName ] = 'free'
        return variable_bounds

    def sampleRandomProcess(self,index):

        self.sample_forecast(index[0:2])
        self.sample_inflows(index[2:4])
        self.random_sample_probability = self.inflowRandomSampleProbability * self.forecastRandomSampleProbability


    def sample_forecast(self,index):
        month_name = MONTH_LIST[self.global_state.currentMonth]
        self.forecastRandomSampleProbability = 1

        if month_name in self.parameters.se_fc_rwn_month:
            self.forecast_sample_value(index)
            for i in range(len(self.parameters.riverName)):
                self.se_fc_rwn[i] = np.sum( self.forecastRandomSampleValue * np.sqrt( self.parameters.forecastEigenValue[self.global_state.currentMonth]) * self.parameters.forecastEigenVector[self.global_state.currentMonth,i] )

        if month_name in self.parameters.se_fc_month:
            if month_name in self.parameters.se_fc_gen_month:
                if month_name == self.parameters.se_fc_start_month:
                    self.se_fc_resi[:] = self.se_fc_rwn
                else:
                    if month_name in self.parameters.se_fc_rwn_month :
                        self.se_fc_resi[:] = self.parameters.forecastRou[self.global_state.currentMonth] * self.previousForecastResidual + np.sqrt( 1 - self.parameters.forecastRou[self.global_state.currentMonth] **2 ) * self.se_fc_rwn
                    else:
                        self.se_fc_resi[:] = self.parameters.forecastRou[self.global_state.currentMonth] * self.previousForecastResidual
            else : 
                self.se_fc_resi[:] = self.previousForecastResidual

            self.forecastResidual[:] = self.se_fc_resi

        if month_name in self.parameters.se_fc_month:
            self.se_fc_reservoir[:] = self.parameters.forecastReservoirIntercept[self.global_state.currentMonth] + self.parameters.forecastReservoirSlope[self.global_state.currentMonth] * (self.yearPlanStart + self.global_state.currentYear -1) + self.parameters.forecastReservoirStd[self.global_state.currentMonth] * self.se_fc_resi

    ### Seasonal inflow forecast of Peace and Columbia
        if month_name in self.parameters.se_fc_month:
            self.se_fc[:] =  self.parameters.forecastIntercept[self.global_state.currentMonth] + self.parameters.forecastSlope[self.global_state.currentMonth]* self.se_fc_reservoir

    ### Seasonal inflow 
        if month_name in self.parameters.se_fc_month:
            self.se_actual = self.parameters.forecastActualIntercept[self.global_state.currentMonth] + self.parameters.forecastActualSlope[self.global_state.currentMonth] * self.se_fc
        else:
            self.se_actual = np.zeros(self.se_actual.size)

        self.forecast[:] = self.se_actual


    def sample_inflows(self,index):
        month_name = MONTH_LIST[self.global_state.currentMonth]
        self.inflowRandomSampleProbability = 1

        self.rwn = np.zeros(len(self.parameters.riverName))
        if month_name in self.parameters.stage_rand_inflow :
            self.inflow_sample_value(index)
            for i in range(len(self.parameters.riverName)):
                self.rwn[i] = math.exp(
                                    np.sum( self.inflowRandomSampleValue * np.sqrt(self.parameters.inflowEigenValue[self.global_state.currentMonth]) * self.parameters.inflowEigenVector[self.global_state.currentMonth,i] ) * self.parameters.inflowRwnStd[self.global_state.currentMonth,i] + self.parameters.inflowRwnMean[self.global_state.currentMonth,i] ) \
                              - self.parameters.inflowRwnShift[self.global_state.currentMonth,i]

        previous_month_name = MONTH_LIST[ self.global_state.currentMonth - 1 ]
        if month_name in self.parameters.stage_rand_inflow and previous_month_name in self.parameters.stage_rand_inflow :
            self.resi_sv[:] = self.previousInflowResidual
        else:
            self.resi_sv[:] = np.zeros(self.resi_sv.shape)

        if month_name in self.parameters.stage_rand_inflow:
            if previous_month_name not in self.parameters.stage_rand_inflow:
                self.resi[:] = self.rwn
            else:
                self.resi[:] = self.parameters.inflowRou[self.global_state.currentMonth] * self.resi_sv + np.sqrt(1-self.parameters.inflowRou[self.global_state.currentMonth]**2) * self.rwn

        self.inflowResidual[:] = self.resi

        self.inflowRiver[:] = self.parameters.inflowIntercept[self.global_state.currentMonth] + self.parameters.inflowSlopeYear[self.global_state.currentMonth] * (self.yearPlanStart + self.global_state.currentYear -1 )
        if month_name in self.parameters.stage_inflow_se:
            self.inflowRiver[:] += self.parameters.inflowSlopeSeasonal[self.global_state.currentMonth] * self.forecast
        if month_name in self.parameters.stage_rand_inflow:
            self.inflowRiver[:] += self.parameters.inflowStd[self.global_state.currentMonth] * self.resi

        inflow_min = 0.
        for i, inflow in enumerate(self.inflowRiver):
            if inflow < inflow_min :
                self.inflow_corrected[i] = inflow_min - inflow
                self.inflowRiver[i] = inflow_min
            else:
                self.inflow_corrected[i] = 0.

        for plantIndex, plantName in enumerate(self.parameters.plantName):
            if plantName in self.parameters.plantInflowVaryName :
                indexInflowPlantVary = self.parameters.plantInflowVaryName.index(plantName)
                for riverIndex,plantsForEachRiver in enumerate(self.parameters.plantRiver):
                    if plantName in plantsForEachRiver:
                        self.inflow[plantIndex] = self.parameters.inflowPercentage[self.global_state.currentMonth,indexInflowPlantVary] * self.inflowRiver[riverIndex] 
                        break
            else:
                indexFixedInflow = self.parameters.plantInflowFixedName.index(plantName)
                self.inflow[plantIndex] = self.parameters.plantInflowFixedValue[indexFixedInflow]

    def forecast_sample_value(self, index):
        for i in range(2):
            if index == []:
                randomNumber = random.random()
                for j,bin_val in enumerate(self.parameters.forecastRvDistCumProb[self.global_state.currentMonth]):
                    if randomNumber <= bin_val:
                        self.forecastRandomSampleValue[i] = self.parameters.forecastRvDistValue[self.global_state.currentMonth , j]
                        self.forecastRandomSampleProbability = self.forecastRandomSampleProbability * self.parameters.forecastRvDistProb[self.global_state.currentMonth ,j]
                        break
            else:
                self.forecastRandomSampleValue[i] = self.parameters.forecastRvDistValue[self.global_state.currentMonth, index[i]]
                self.forecastRandomSampleProbability = self.forecastRandomSampleProbability * self.parameters.forecastRvDistProb[self.global_state.currentMonth, index[i]]

    def inflow_sample_value(self, index):
        for i in range(2):
            if index == []:
                randomNumber = random.random()
                for j,bin_val in enumerate(self.parameters.inflowDistCumProb[self.global_state.currentMonth]):
                    if randomNumber <= bin_val:
                        self.inflowRandomSampleValue[i] = self.parameters.inflowDistValue[self.global_state.currentMonth , j]
                        self.inflowRandomSampleProbability = self.inflowRandomSampleProbability * self.parameters.inflowDistProb[self.global_state.currentMonth ,j]
                        break
            else:
                self.inflowRandomSampleValue[i] = self.parameters.inflowDistValue[self.global_state.currentMonth, index[i]]
                self.inflowRandomSampleProbability = self.inflowRandomSampleProbability * self.parameters.inflowDistProb[self.global_state.currentMonth, index[i]]


    def applyTransition(self, decision, variableList=[]):
        self.set_state_value([self.forecastResidual, self.inflowResidual])

    # self.forecast = self.inflowModel.inflowRiverCalculation( self.timeStep , t_now, self.currentYear )
    ########################################
    # Seasonal inflow Forecast Calculation #
    ########################################

##############################################
### Inflow Model Parameters Class ####
##############################################
class inflowModelParameters(object):
    def __init__(self, path):
        # TODO : move to system class
        self.riverName = ['peace', 'columbia' ]
        self.plantName = ['GMS', 'PCN', 'STC', 'MCA', 'REV', 'ARD']
        self.plantInflowVaryName = ['GMS' , 'STC' ,'MCA' , 'REV' , 'ARD']
        self.plantInflowFixedName = ['PCN']
        self.plantInflowFixedValue = [ 20 ]
        self.plantRiver = [['GMS', 'PCN', 'STC'] , ['MCA', 'REV', 'ARD']]

        ###############################
        # Stochastic Forecast Model
        ################################
        self.se_fc_rwn_month = ['Jan','Feb']
        self.se_fc_month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']
        self.se_fc_gen_month = ['Jan','Feb','Mar','Apr']
        self.se_fc_start_month = 'Jan'

        self.monthForecastGenerationName = ['Jan','Feb','Mar','Apr']
        self.isMonthForecastGeneration = self.setIsMonthInList(self.monthForecastGenerationName) # return Binary
        self.monthStartForecastGenerationName = ['Jan']
        self.isMonthStartForecastGeneration = self.setIsMonthInList(self.monthStartForecastGenerationName)
        self.monthWhiteNoiseForecastGenerationName = ['Jan','Feb']
        self.isMonthWhiteNoiseForecastGeneration = self.setIsMonthInList(self.monthWhiteNoiseForecastGenerationName)
        self.nBinForecast = 3
        self.setForecastReservoirIntercept()
        self.setForecastReservoirSlope()
        self.setForecastReservoirStd()
        self.setForecastIntercept()
        self.setForecastSlope()
        self.setForecastActualIntercept()
        self.setForecastActualSlope()
        self.setForecastRou()
        self.setForecastEigenValue()
        self.setForecastEigenVector()
        self.setForecastRvDistValue()
        self.setForecastRvDistProb()

        ###############################
        # Inflow Model
        ################################
        self.stage_rand_inflow = ['Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']
        self.stage_inflow_se = ['Apr','May','Jun','Jul','Aug']
        self.monthMeanInflow = ['Dec','Jan','Feb','Mar']
        self.isStochasticInflows = 1 - self.setIsMonthInList( self.monthMeanInflow ).astype('int8')
        self.monthInflowSeasonalForecast = ['Apr','May','Jun','Jul','Aug']
        self.isInflowSeasonalForecast = self.setIsMonthInList(self.monthInflowSeasonalForecast)
        self.setInflowPercentage()
        self.setInflowIntercept()
        self.setInflowSlopeYear()
        self.setInflowSlopeSeasonal()
        self.setInflowStd()
        self.setInflowRou()
        self.setInflowRou()
        self.setInflowWhiteNoise()
        self.setInflowEigenValue()
        self.setInflowEigenVector()
        self.setInflowDistValue()
        self.setInflowDistProb()


        ###############################
        # Stochastic Forecast Model
        ################################


    # Set Forecast Parameters
    def setForecastReservoirIntercept(self):
        param = np.array([[1865070,1202580], 
                        [1484559,952578],
                        [1045143,740061],
                        [791622,492507]])
        self.forecastReservoirIntercept = self.setMonthParameters( param , self.monthForecastGenerationName)

    def setForecastReservoirSlope(self):
        param = np.array([[-780.4530,-522.1890],
                            [-588.7440,-397.1880],
                            [-370.5660,-291.7710],
                            [-241.2810,-167.5350]])
        self.forecastReservoirSlope = self.setMonthParameters(param, self.monthForecastGenerationName)

    def setForecastReservoirStd(self):
        param = np.array([[26989,12837],
                            [27617,13250],
                            [33446,15071],
                            [31885,14734]])
        self.forecastReservoirStd = self.setMonthParameters(param, self.monthForecastGenerationName)


    def setForecastIntercept(self):
        param = np.array([[0,-8225],[0,-8065],[0,-16406],[0,-14737]])
        self.forecastIntercept = self.setMonthParameters(param , self.monthForecastGenerationName)

    def setForecastSlope(self):
        param = np.array([[1.0000,2.2319],
                            [1.0000,2.2325],
                            [1.0000,2.2832],
                            [1.0000,2.2750]])
        self.forecastSlope = self.setMonthParameters(param , self.monthForecastGenerationName)

    def setForecastActualIntercept(self):
        param = np.array([[148081,24153],
                            [56477,3628],
                            [82386,8710],
                            [8560,-7502]])
        self.forecastActualIntercept = self.setMonthParameters(param , self.monthForecastGenerationName)

    def setForecastActualSlope(self):
        param = np.array([[0.7160,0.8412],
                            [1.0100,0.8975],
                            [0.9150,0.8871],
                            [1.1330,0.9314]])
        self.forecastActualSlope = self.setMonthParameters(param , self.monthForecastGenerationName)

    def setForecastRou(self):
        param = np.array([[1.00,1.00],
                        [0.84,0.85],
                        [0.93,0.95],
                        [0.96,0.95]])
        self.forecastRou = self.setMonthParameters(param , self.monthForecastGenerationName)

    def setForecastEigenValue(self):
        param= np.array([[1.554,0.446],
                        [1.554,0.446]])
        self.forecastEigenValue = self.setMonthParameters(param , self.monthWhiteNoiseForecastGenerationName)

    def setForecastEigenVector(self):
        self.forecastEigenVector = np.empty( [len(MONTH_LIST) , len(self.riverName) ,  2 ] )
        param = np.array([[0.707,-0.707],
                           [0.707,0.707]])
        for month in self.monthWhiteNoiseForecastGenerationName:
            self.forecastEigenVector[MONTH_LIST.index(month)] = param # peace param
            

    def setForecastRvDistValue(self):
        self.forecastRvDistValue = np.empty( [len(MONTH_LIST) , self.nBinForecast ] )
        # peace param
        param = np.array([[-1,0,1],
                            [-1,0,1]])
        for i,month in enumerate(self.monthWhiteNoiseForecastGenerationName):
            self.forecastRvDistValue[MONTH_LIST.index(month)] = param[i]

    def setForecastRvDistProb(self):
        self.forecastRvDistProb = np.empty( [len(MONTH_LIST) , self.nBinForecast ] )
        self.forecastRvDistCumProb = np.empty( [len(MONTH_LIST) , self.nBinForecast ] )

        param = np.array([[0.309,0.382,0.309],
                            [0.309,0.382,0.309]])
        for i,month in enumerate(self.monthWhiteNoiseForecastGenerationName):
            self.forecastRvDistProb[MONTH_LIST.index(month)] = param[i]
            self.forecastRvDistCumProb[MONTH_LIST.index(month)] = np.cumsum(param[i])


    #################################
    # Stochastic Inflow model 
    #################################

    def setInflowPercentage(self):
        param = np.array([[0.948,0.052,0.443,0.159,0.398],
                    [0.949,0.051,0.446,0.160,0.395],
                    [0.948,0.052,0.410,0.171,0.420],
                    [0.928,0.072,0.360,0.201,0.440],
                    [0.925,0.075,0.397,0.224,0.379],
                    [0.919,0.081,0.488,0.216,0.296],
                    [0.894,0.106,0.567,0.198,0.235],
                    [0.896,0.104,0.616,0.185,0.198],
                    [0.905,0.095,0.582,0.193,0.224],
                    [0.928,0.072,0.505,0.212,0.283],
                    [0.942,0.058,0.449,0.217,0.334],
                    [0.941,0.059,0.432,0.189,0.378]    ])
        self.inflowPercentage = param

    def setInflowIntercept(self):
        param = np.array([[-3853,-1849],
                        [-2754, -432],
                        [-2303,-2456],
                        [-10402,-6836],
                        [-18046,-1594],
                        [969, 1782],
                        [15318,-8389],
                        [11857,14874],
                        [5041,13235],
                        [3593, 2172],
                        [-1348, 1175],
                        [-1807,  372]])
        self.inflowIntercept = param

    def setInflowSlopeYear(self):
        param = np.array([[2.1029,1.0653],
            [1.5283,0.3401],
            [1.3016,1.3810],
            [5.4819,3.6471],
            [9.7905,1.3579],
            [-0.9291,-1.0157],
            [-8.1081,3.6276],
            [-6.0884,-7.5319],
            [-2.0849,-6.1713],
            [-1.3676,-0.7807],
            [0.9666,-0.3631],
            [1.0992,-0.0287]])
        self.inflowSlopeYear = param

    def setInflowSlopeSeasonal(self):
        param = np.array([[0.000369,0.000763],
                [0.003953,0.002797],
                [0.015380,0.010957],
                [0.009056,0.012358],
                [0.004007,0.005761]])
        self.inflowSlopeSeasonal = self.setMonthParameters( param , self.monthInflowSeasonalForecast)

    def setInflowStd(self):
         param = np.array([[64.2,44.9],
                         [48.9,50.1],
                         [61.1,65.6],
                         [218.3,191.2],
                         [646.3,388.4],
                         [656.1,408.6],
                         [436.3,295.6],
                         [283.3,275.4],
                         [249.5,239.8],
                         [250.3,192.6],
                         [130.7,136.0],
                         [63.6,66.8]])
         self.inflowStd = param

    def setInflowRou(self):
        param = np.array([[0,0.25,-0.58,-0.11,0.41,0.34,0.53,0.61],
                            [0,0.29,-0.43,-0.08,0.28,0.41,0.38,0.47]])
        param = np.transpose(param)
        self.inflowRou = np.zeros( [ len(MONTH_LIST), len(self.riverName) ] )
        self.inflowRou[self.isStochasticInflows.astype('bool_')] = param

    def setInflowWhiteNoise(self):
        self.inflowRwnShift = np.zeros([ len(MONTH_LIST) , len(self.riverName)] )
        self.inflowRwnMean = np.zeros([ len(MONTH_LIST) , len(self.riverName)] )
        self.inflowRwnStd = np.zeros([ len(MONTH_LIST) , len(self.riverName)] )
        for month in ['Apr','May','Jun','Jul','Aug']:
            self.inflowRwnShift[MONTH_LIST.index(month)] = [8.10,26.85]    
            self.inflowRwnMean[MONTH_LIST.index(month)] = [2.084,3.290]    
            self.inflowRwnStd[MONTH_LIST.index(month)]    = [0.123,0.037]
        for month in ['Sep','Oct','Nov']:
            self.inflowRwnShift[MONTH_LIST.index(month)] = [12.30,3.00]    
            self.inflowRwnMean[MONTH_LIST.index(month)] = [2.506,1.048]    
            self.inflowRwnStd[MONTH_LIST.index(month)]    = [0.081,0.315]

    def setInflowEigenValue(self):
        self.nInflowEigenValue = 2
        self.inflowEigenValue = np.zeros([ len(MONTH_LIST) , self.nInflowEigenValue] )
        for month in ['Apr','May','Jun','Jul','Aug']:
            self.inflowEigenValue[MONTH_LIST.index(month)] = [1.497,0.503]
        for month in ['Sep','Oct','Nov']:
            self.inflowEigenValue[MONTH_LIST.index(month)] = [1.216,0.784]    

    def setInflowEigenVector(self):
        self.inflowEigenVector = np.empty( [ len(MONTH_LIST) , len(self.riverName) , 2 ] )
        # peace param
        param = np.array([[0.707,-0.707],[0.707,0.707]])
        for month in ['Apr','May','Jun','Jul','Aug']:
            self.inflowEigenVector[MONTH_LIST.index(month)] = param # peace
        for month in ['Sep','Oct','Nov']:
            self.inflowEigenVector[MONTH_LIST.index(month)] = param # peace

    def setInflowDistValue(self):
        self.nInflowDistValue = 3
        self.inflowDistValue = np.empty([ len(MONTH_LIST) , self.nInflowDistValue] )
        for month in ['Apr','May','Jun','Jul','Aug']:
            self.inflowDistValue[MONTH_LIST.index(month)] = [-1.475,0,1.475]
        for month in ['Sep','Oct','Nov']:
            self.inflowDistValue[MONTH_LIST.index(month)] = [-1.475,0,1.475]

    def setInflowDistProb(self):
        self.inflowDistProb = np.empty([ len(MONTH_LIST) , self.nInflowDistValue] )
        self.inflowDistCumProb = np.empty([ len(MONTH_LIST) , self.nInflowDistValue] )

        for month in ['Apr','May','Jun','Jul','Aug']:
            self.inflowDistProb[MONTH_LIST.index(month)] = [0.23,0.54,0.23]
            self.inflowDistCumProb[MONTH_LIST.index(month)] = np.cumsum(self.inflowDistProb[MONTH_LIST.index(month)])
        for month in ['Sep','Oct','Nov']:
            self.inflowDistProb[MONTH_LIST.index(month)] = [0.23,0.54,0.23]
            self.inflowDistCumProb[MONTH_LIST.index(month)] = np.cumsum(self.inflowDistProb[MONTH_LIST.index(month)])


############### useful methods
    def setIsMonthInList(self, activeMonthList ):
        isInList = np.zeros( len(MONTH_LIST) )
        for i, month in enumerate(activeMonthList):
            isInList[MONTH_LIST.index(month)] = 1
        return isInList

    def setMonthParameters(self, param , activeMonth ):
        monthParam = np.empty(  [ len(MONTH_LIST) , len(self.riverName) ] )
        for month in MONTH_LIST:
            if month in activeMonth:
                monthParam[MONTH_LIST.index(month)] = param[activeMonth.index(month)]
            else:
                monthParam[MONTH_LIST.index(month)] = param[-1]
        return monthParam


