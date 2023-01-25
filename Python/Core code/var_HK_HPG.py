import numpy as np

from readFileTest import *
from solver import linear_constraint, objective, concave_piecewise_linear_contraint
from model import Module

MONTH_LIST = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


class Var_HK_HPG(Module):
    """docstring for var_HK_HPG"""
    def __init__(self, path):
        super(Var_HK_HPG, self).__init__()
        self.type = 'hpg'

        self.loadParameters(path)
        self.P_bkpt = np.zeros([len(self.params.plant_HK_s), np.amax(self.params.HPG_piece_n)])
        self.QT_bkpt = np.zeros([len(self.params.plant_HK_s), np.amax(self.params.HPG_piece_n)])
        ### HPG curve, i.e., power from turbine release given FB
        self.V_ave_HPG = np.zeros(len(self.params.plant_HK_s))
        self.FB = np.zeros(len(self.params.plant_HK_s))

        self.recall_var = [plant+ '_storage_next_ts' for plant in self.params.plant_HK_s]

        self.var_hk_rhs = np.zeros(len(self.params.plant_HK_s))
        self.production_pwl_coeff = [ np.zeros( [len(self.P_bkpt[i])-1, 2]) for i,plant in enumerate(self.params.plant_HK_s) ]
        self.production_pwl_rhs = [ np.zeros( len(self.P_bkpt[i])-1 ) for i,plant in enumerate(self.params.plant_HK_s) ]

    def updateValues(self, end_storage=[]):
        t_now = self.global_state.currentMonth
        month_name = MONTH_LIST[self.global_state.currentMonth]
        
        for i,plant in enumerate(self.params.plant_HK_s):
            indexPlant = self.params.plantName.index(plant)
            if end_storage == []:
                V_now_HPG = self.params.V_now_HPG_intercept[t_now,i] + self.params.V_now_HPG_slope_initial[t_now,i] * self.global_state.storage[indexPlant] \
                                    + self.params.V_now_HPG_slope_inflow[t_now,i] * self.random_variable.inflows[indexPlant]
                if month_name in self.params.fcc_stage and plant in self.params.fccPlantName:
                    V_now_HPG = min(V_now_HPG , self.params.VMax1[indexPlant] - self.global_state.fcc_1[self.params.fccPlantName.index(plant)] )
                else:
                    V_now_HPG = min(V_now_HPG,self.params.VMax1[indexPlant])

            else:
                V_now_HPG = end_storage[i]
            self.V_ave_HPG[i] = (self.global_state.storage[indexPlant] + V_now_HPG) / 2

            if plant =='MCA':
                self.V_ave_HPG[i] = self.V_ave_HPG[i] + self.params.V_NT_BC_max[t_now] - self.global_state.V_NT_BC_value 

            # linear interpolation into self.params.V_bkpy and FB_bkpt #TODO : BREAK OR WHILE LOOP
            for j in range(len(self.params.V_bkpt[i])):
                if self.V_ave_HPG[i] < self.params.V_bkpt[i][j]:
                    alpha = (self.params.V_bkpt[i][j]-self.V_ave_HPG[i]) / (self.params.V_bkpt[i][j]-self.params.V_bkpt[i][j-1])
                    self.FB[i] = alpha * self.params.FB_bkpt[i][j-1] + (1-alpha) * self.params.FB_bkpt[i][j]
                    break


        for j in range(len(self.params.plant_HK_s)):
            self.P_bkpt[j] = self.params.P_bkpt_slope[j] * self.FB[j] + self.params.P_bkpt_intercept[j]
            self.QT_bkpt[j] = self.params.QT_bkpt_slope[j] * self.FB[j] + self.params.QT_bkpt_intercept[j]

        # hk production
        for j, plant in enumerate(self.params.plantName):
            if plant in self.params.plant_HK_s:
                idx = self.params.plant_HK_s.index(plant)
                if plant == 'MCA':
                    # Plant_generation_var_HK_head_loss
                    self.var_hk_rhs[idx] = self.params.P_adj[idx] * (self.V_ave_HPG[idx] - self.params.V_NT_BC_max[t_now])
                else:
                    self.var_hk_rhs[idx] = 0. #self.params.P_adj[idx] * self.V_ave_HPG[idx]

                # pwl max produ
                for i in range(len(self.P_bkpt[idx]) - 1):
                    slope = (self.P_bkpt[idx][i + 1] - self.P_bkpt[idx][i]) / (self.QT_bkpt[idx][i + 1] - self.QT_bkpt[idx][i])
                    self.production_pwl_coeff[idx][i][:] = [- slope, 1.]
                    self.production_pwl_rhs[idx][i] = self.P_bkpt[idx][i] - slope * self.QT_bkpt[idx][i]

    def get_constraints(self):#, current_month):
        t_now = self.global_state.currentMonth

        self.updateValues()
        constraints = []

        ### Power generation using HPG curves
        for i,h in enumerate(self.params.hplName):
            for j,plant in enumerate(self.params.plantName):
                if plant not in self.params.plant_HK_s:
                    # fix HK
                    constraints.append( 
                        linear_constraint( [ 'production_' + plant +'_'+ h , 'QT_' + plant +'_'+ h ] , [1, - self.params.HK[t_now,j] ] ,'E', 0.))
                else:
                    idx = self.params.plant_HK_s.index(plant)
                    variable_name = ['QT_' + plant +'_'+ h, 'production_' + plant +'_'+ h + '_pwl']
                    nparts = self.params.HPG_piece_n[idx] - 1
                    sense = 'L' #* nparts
                    for i in range(nparts):
                        constraints.append(
                            linear_constraint( variable_name, self.production_pwl_coeff[idx][i][0:2] , sense, self.production_pwl_rhs[idx][i:i+1] ))
                    if plant == 'MCA':
                        # Plant_generation_var_HK_head_loss 
                        #constraints.append(
                        #    linear_constraint(['production_' + plant +'_'+ h , 'production_' + plant +'_'+ h + '_pwl' , plant + '_storage_average' , 'V_NT_BC_sv' , 'V_NT_BC_now' ],
                        #            [1., - 1.] + [self.params.P_adj[idx] ,- self.params.P_adj[idx] /2 , - self.params.P_adj[idx]/2]  , 'L' , self.var_hk_rhs[idx:idx+1], is_dynamic={'cases':["state","random_process","var_HK"],'sides':'rhs'} ) )
                        constraints.append(
                            linear_constraint(['production_' + plant +'_'+ h , 'production_' + plant +'_'+ h + '_pwl' , plant + '_storage_average' , 'V_NT_BC_sv' , 'V_NT_BC_now' ],
                                    [1., - 1., self.params.P_adj[idx] ] + [- self.params.P_adj[idx] /2 , - self.params.P_adj[idx]/2]  , 'L' , self.var_hk_rhs[idx:idx+1] ) )
                    else:
                        # Plant_generation_var_HK
                        constraints.append( 
                            linear_constraint(
                                ['production_' + plant + '_' + h, 'production_' + plant + '_' + h + '_pwl',
                                 plant + '_storage_average'],
                                [ 1., - 1. , self.params.P_adj[idx] ]  , 'L' , self.var_hk_rhs[idx:idx+1]) )
                        # linear_constraint( [ 'production_' + plant +'_'+ h , 'production_' + plant +'_'+ h + '_pwl' , plant + '_storage_average' ],
                        #        [ 1., - 1. , self.params.P_adj[idx] ]  , 'L' , self.var_hk_rhs[idx:idx+1], is_dynamic={'cases':["state","random_process","var_HK"],'sides':'both'} ) )

        return constraints

    def loadParameters(self,path):
        # plants using HPG curves
        self.params.plant_HK_s = ['GMS', 'MCA'] 
        # forebay from storage, i.e., FB from V
        self.params.FB_piece_n = [34 , 157 ]    
        # HPG curve, i.e., power from turbine release given FB
        self.params.HPG_piece_n = np.array([10, 10])
        # ?
        self.params.P_adj = [ -4.43385826E-04,-1.97621717E-03]
        # load parameters
        dataRaw = readFileOfTables( path )
        self.params.FB_V_bkpt_GMS = np.array(dataRaw['FB_V_bkpt_GMS'].value, dtype='float')
        self.params.FB_V_bkpt_MCA = np.array(dataRaw['FB_V_bkpt_MCA'].value, dtype='float')
        self.params.P_bkpt_slope = indexing(dataRaw['P_bkpt_slope'],self.params.plant_HK_s)
        self.params.P_bkpt_intercept = indexing(dataRaw['P_bkpt_intercept'],self.params.plant_HK_s)
        self.params.QT_bkpt_slope = indexing(dataRaw['QT_bkpt_slope'],self.params.plant_HK_s)
        self.params.QT_bkpt_intercept = indexing(dataRaw['QT_bkpt_intercept'],self.params.plant_HK_s)
        self.params.V_now_HPG_intercept = indexing(dataRaw['V_now_HPG_intercept_month'],MONTH_LIST)
        self.params.V_now_HPG_slope_initial = indexing(dataRaw['V_now_HPG_slope_initial_month'],MONTH_LIST)
        self.params.V_now_HPG_slope_inflow = indexing(dataRaw['V_now_HPG_slope_inflow_month'],MONTH_LIST)

        #TODO : declare
#        self.params.FB_bkpt = np.zeros([len(self.params.plant_HK_s),max(self.params.FB_piece_n)])
#        self.params.V_bkpt = np.zeros([len(self.params.plant_HK_s),max(self.params.FB_piece_n)])
        self.params.FB_bkpt = [self.params.FB_V_bkpt_GMS[:,0] , self.params.FB_V_bkpt_MCA[:,0]]
        self.params.V_bkpt = [self.params.FB_V_bkpt_GMS[:,1] , self.params.FB_V_bkpt_MCA[:,1]]
            #for j in range(self.params.FB_piece_n[i]-1):
            #    self.params.V_slope[i,j] = (FB_bkpt[j+1] - FB_bkpt[j]) / (V_bkpt[j+1] - V_bkpt[j])
            #self.params.V_intercept[i] = V_bkpt[i,1] - FB_bkpt[i,1] / self.params.V_slope[i,1]


