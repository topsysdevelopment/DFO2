import numpy as np
from model import Module


class Price(Module):
    """docstring for price"""
    def __init__(self):
        super(Price, self).__init__()
        self.type = 'price'

        self.loadParameters()
        self.definePriceExpImp()

    def definePriceExpImp(self):
        self.params.price_Exp_npc = 11
        self.params.price_Exp_rate_end = 0.8

        self.params.price_Exp_bkpt = np.zeros(self.params.price_Exp_npc-1)
        for i in range(self.params.price_Exp_npc-1):
            self.params.price_Exp_bkpt[i]  = float(i+1) / self.params.price_Exp_npc
        self.params.price_Exp_rate = np.zeros(self.params.price_Exp_npc)
        for i in range(self.params.price_Exp_npc):
            self.params.price_Exp_rate[i]  = 1 - (1-self.params.price_Exp_rate_end) / (self.params.price_Exp_npc-1) * (float(i))


        self.params.price_Imp_npc = 3
        self.params.price_Imp_rate_end = 1.02
        self.params.price_Imp_bkpt = np.zeros(self.params.price_Imp_npc-1)
        for i in range(self.params.price_Imp_npc-1):
            self.params.price_Imp_bkpt[i]  = 1 - float(i+1) / self.params.price_Imp_npc
        self.params.price_Imp_rate = np.zeros(self.params.price_Imp_npc)
        for i in range(self.params.price_Imp_npc):
            self.params.price_Imp_rate[i]  = self.params.price_Imp_rate_end - float(self.params.price_Imp_rate_end-1) / (self.params.price_Imp_npc-1) * (i)

    #TODO method with PATH
    def loadParameters(self):
        self.params.price_mp_linear = np.transpose(np.array([[-0.00998,-0.01325,-0.00944,-0.01074,-0.00912,-0.01316,-0.01119,-0.01013,-0.01077,-0.00513,-0.00672,-0.01370],
                                        [-0.00998,-0.01325,-0.00944,-0.01074,-0.00912,-0.01316,-0.01119,-0.01013,-0.01077,-0.00513,-0.00672,-0.01370],
                                        [-0.00998,-0.01325,-0.00944,-0.01074,-0.00912,-0.01316,-0.01119,-0.01013,-0.01077,-0.00513,-0.00672,-0.01370],
                                        [-0.01137,-0.01434,-0.00888,-0.01555,-0.01551,-0.03869,-0.01621,-0.01314,-0.01018,-0.00684,-0.00912,-0.01533],
                                        [-0.01137,-0.01434,-0.00888,-0.01555,-0.01551,-0.03869,-0.01621,-0.01314,-0.01018,-0.00684,-0.00912    -0.01533]]) )

        self.params.price_mp_intercept = np.transpose(np.array(
                                        [[1.9131,2.2124,1.8638,1.9824,1.8346,2.2034,2.0237,1.9265,1.9850,1.4693,1.6143,2.2534],
                                        [1.9131,2.2124,1.8638,1.9824,1.8346,2.2034,2.0237,1.9265,1.9850,1.4693,1.6143,2.2534],
                                        [1.9131,2.2124,1.8638,1.9824,1.8346,2.2034,2.0237,1.9265,1.9850,1.4693,1.6143,2.2534],
                                        [2.0395,2.3113,1.8121,2.4225,2.4190,4.5387,2.4828,2.2020,1.9315,1.6254,1.8345,2.4026],
                                        [2.0395,2.3113,1.8121,2.4225,2.4190,4.5387,2.4828,2.2020,1.9315,1.6254,1.8345,2.4026]] ) )
