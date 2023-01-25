import random
import mpi4py.MPI as MPI

from sddp import SDDP
from cfdp import CFDP

from model import Model
from inflow import Inflow
from system import System
from treaty import Treaty
from var_HK_HPG import Var_HK_HPG
from finalModule import FinalModule
from price import Price


__version__ = "1.0"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
random.seed(13 + rank)

##############################################
### Parameters #### # TODO : move in parameters file
##############################################

solver = 'Lp_solve' #'Cplex'#

inflowPath = './inflowModel/'
dataPath = './Data/Case_study.data'
treatyPath = './Data/treatyData.txt'
varHkPath = './Data/var_HK_HPG.txt'
stateVariable = ['GMS_storage_sv', 'V_TY_total_sv', 'V_TY_MCA_sv', 'V_NT_BC_sv'] \
                + [ "resi_sv_peace" , "resi_sv_columbia" ] \
                + [ "previous_forecast_residual_sv_peace" , "previous_forecast_residual_sv_columbia" ]  # state variable names
##############################################
### Instantiate objects ###
##############################################
# TODO : module declaration does not depend on other module
inflowModel = Inflow(inflowPath)
system = System(dataPath, inflowModel)
treaty = Treaty(treatyPath, system)
varHK = Var_HK_HPG(varHkPath)
price = Price()
finalModule = FinalModule(system)


model = Model((inflowModel, system, treaty, varHK, finalModule))

sample_mode = 'unique' #'multi_random'#
nCuts = 200 # number forward passes
algo_optim = SDDP(model, stateVariable, solver, sample_mode)

# optimize and simulate every 10 cuts
algo_optim.optimize(nCuts, n_simulate = 10)
