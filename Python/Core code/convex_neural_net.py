from math import sqrt, log, factorial
import tensorflow as tf
import pickle
#import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()


import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import itertools

from solver import Linear_problem, objective, linear_constraint
from model import Variable
from water_value import WvFunction


class Convex_NN(WvFunction):
    def __init__(self, next_state_variable, name = "", fitting_mode = True):

        self.next_state_variable = next_state_variable
        self.name = name
        ##################################
        ## neural network
        self.dataX = []
        self.dataDx = []
        self.dataY = [] #dataY.reshape((-1, 1)).astype(np.float32)
        self.data_raw_X = []
        self.data_raw_Dx = []
        self.data_raw_Y = []

        self.data_training = []

        #self.nData = dataX.shape[0]
        self.nFeatures = len(next_state_variable)
        self.nLabels = 1
        self.nEpoch = 5
        
        self.n_hidden_sz = [20, 1]
        self.learning_rate = 0.01 
        self.lr_gloabl_step = 25
        self.lr_decay_steps = 0.98

        self.n_hidden_sz_input = [self.nFeatures] + self.n_hidden_sz
        # start tf session
        self.config = tf.ConfigProto(log_device_placement=False)
        self.graph = tf.Graph()
        self.sess = tf.Session(config=self.config, graph = self.graph) 

        self.fitting_mode = fitting_mode

        self.iteration  = 0
        self.create_nn()

        self.update_standardized_value = True
        self.initialize()
        
        # variable to standardize values
        self.std_x = np.ones(len(next_state_variable))
        self.mean_x = np.zeros(len(next_state_variable))

        self.std_y = np.ones(1 )
        self.mean_y = np.zeros( 1 )

        #############################
        # declare constraint variables
        self.rhs_norm_x = np.zeros(self.nFeatures)
        self.coeff_norm_x = np.ones([self.nFeatures,2])
        self.rhs_layer = [ np.zeros( (n_nodes,1) )  for n_nodes in self.n_hidden_sz]
        self.w_nn = []
        self.alpha_prelu = [ [] for _ in self.n_hidden_sz[:-1] ]
        
    def get_variable_bounds(self):
        return self.var_bound

    def get_nn_var(self):
        nn_var= []
        cons = self.get_constraints()
        for c in cons:
            for var in c.variable_name:
                if 'z' in var:
                    nn_var.append(var)
        return list(set(nn_var))


    def make_variable_bound(self):
        var_bound= {}
        for var in self.nn_var:
            var_bound[var] = 'free'
        return var_bound

       
    def train(self, iteration = True):
        if self.update_standardized_value:
            self.update_norm_params()
        self.dataX, self.dataDx, self.dataY = self.normalize_data(self.data_raw_X, self.data_raw_Dx, self.data_raw_Y)

        for i in range(self.nEpoch):
            summary, _, trainMSE, yn, dxn = self.sess.run(
                [self.merged, self.train_step, self.loss_function, self.y_, self.dx_],
                feed_dict={self.x_: self.dataX, self.TrueDx_: self.dataDx, self.trueY_: self.dataY})
            if len(self.proj) > 0:
                self.sess.run(self.proj)
        if iteration is True:
            self.train_writer.add_summary(summary, self.iteration)
            self.iteration += 1


    def create_nn(self):
        with self.graph.as_default():
            # declare variable for tf
            self.trueY_ = tf.placeholder(tf.float32, shape=[None, 1], name='trueY')
            self.TrueDx_ = tf.placeholder(tf.float32, shape=[None, self.nFeatures], name='trueDx')

            self.y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y')
            self.dx_ = tf.placeholder(tf.float32, shape=[None, self.nFeatures], name='dx')
            self.x_ = tf.placeholder(tf.float32, shape=[None, self.nFeatures], name='x')
    
            # loss function
            self.y_ = self.nn_design(self.x_)
            self.dx_ = tf.squeeze(tf.gradients(self.y_, self.x_),[0])

            step = tf.Variable(0, trainable = False)
            self.lr = tf.train.exponential_decay(self.learning_rate, step, self.lr_gloabl_step, self.lr_decay_steps )
            #lr = self.learning_rate

            self.loss_function = ( tf.losses.mean_squared_error(self.TrueDx_, self.dx_, weights= 1.0) )
                                #+ 10e-4 * tf.losses.mean_squared_error(self.trueY_, self.y_, weights = 1.0) )
            
            self.train_step = ( #tf.train.AdamOptimizer(self.lr)
                                tf.train.RMSPropOptimizer(self.lr)
                            .minimize(self.loss_function, global_step = step) )

            #self.opt = tf.train.AdamOptimizer(self.learning_rate)
            self.theta_ = tf.trainable_variables()
            self.all_var_ = tf.global_variables()
            #self.gv_ = [(g,v) for g,v in
            #            self.opt.compute_gradients(self.loss_function, self.theta_)
            #            if g is not None]
            #self.train_step = self.opt.apply_gradients(self.gv_)

            self.theta_cvx_ = [v for v in self.theta_
                               if 'proj' in v.name and 'weights:' in v.name]
    
            self.makeCvx = [v.assign(tf.abs(v)/10.) for v in self.theta_cvx_]
            self.proj = [v.assign(tf.maximum(v, 0)) for v in self.theta_cvx_]
    
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.makeCvx) 
            tf.summary.scalar('MSE' + self.name, self.loss_function)
            for i in range(self.nFeatures):
                tf.summary.histogram('X'+ str(i)+ '_' + self.name, self.x_[:,i] )
            tf.summary.histogram('Y_' + self.name, self.trueY_)
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter('./train/' + 'discount_scen_for_y_' +self.name + '/', self.graph, filename_suffix = '')




    def evaluate(self, x):
        # normalize
        x_norm = self.normalize_data(x)
        y_norm = self.sess.run(self.y_,feed_dict={self.x_: x_norm})
        return self.un_normalize_y(y_norm)

    def plot_prediction(self, id_x = 0):

        fig = plt.figure()      
        if self.dataX.shape[1] == 1 :
            y_eval = self.evaluate( self.data_raw_X )
            plt.scatter(self.data_raw_X[:,id_x], y_eval, color='r')#, alpha = 0.7, linestyle='dashed',lw = 5)
            plt.scatter(self.data_raw_X[:,id_x], self.data_raw_Y, color='b')
        elif self.dataX.shape[1] == 2 :
            ax = fig.add_subplot(111, projection = '3d')
            x_eval = np.mgrid[-4:4:0.5, -4:4.:0.5].reshape(2,-1).T
            y_eval = self.sess.run(self.y_,feed_dict={self.x_: x_eval})
            n_resize = int(sqrt(len(x_eval)))
            xx_eval = x_eval[:,0].reshape((n_resize,n_resize))
            xy_eval = x_eval[:,1].reshape((n_resize,n_resize))
            y_eval = y_eval.reshape((n_resize,n_resize))

            #ax.plot_surface(xx_eval,xy_eval,y_eval)
            ax.scatter(self.data_raw_X[:,0], self.data_raw_X[:,1], self.data_raw_Y, color='green')

            data_Y_constraint = []
            for x in self.data_raw_X:
                data_Y_constraint.append( self.evaluate_constraint(x) )
            ax.scatter(self.data_raw_X[:,0], self.data_raw_X[:,1], data_Y_constraint, color='red')
        
        #plt.show()


    def plot_surface(self, x):
        fig = plt.figure()      
        y_eval = self.sess.run(self.y_,feed_dict={self.x_: self.dataX})
        if self.dataX.shape[1] == 1 :
            plt.scatter(self.dataX[:,id_x], y_eval, color='r')#, alpha = 0.7, linestyle='dashed',lw = 5)
            plt.scatter(self.dataX[:,id_x], self.dataY, color='b')
        elif self.dataX.shape[1] == 2 :
            ax = fig.add_subplot(111, projection = '3d')
            x_eval = np.mgrid[-4:4:0.5, -4:4.:0.5].reshape(2,-1).T
            y_eval = self.sess.run(self.y_,feed_dict={self.x_: x_eval})
            n_resize = int(sqrt(len(x_eval)))
            xx_eval = x_eval[:,0].reshape((n_resize,n_resize))
            xy_eval = x_eval[:,1].reshape((n_resize,n_resize))
            y_eval = y_eval.reshape((n_resize,n_resize))

            #ax.plot_surface(xx_eval,xy_eval,y_eval)
            ax.scatter(self.dataX[:,0], self.dataX[:,1], self.dataY, color='green')

            #data_Y_constraint = []
            #for x in self.dataX:
            #    data_Y_constraint.append( self.evaluate_constraint(x) )
                    
            #ax.scatter(self.data_raw_X[:,0], self.data_raw_X[:,1], data_Y_constraint, color='red')
        plt.show()


    def initialize(self, initialization_mode = ''):
        n_initial_data = 100     

        self.data_raw_X = np.random.sample( (n_initial_data, self.nFeatures) ) * 2. - 1.
        if initialization_mode == 'x2' or initialization_mode == '' :
            self.data_raw_Y = np.sum( (self.data_raw_X ) * (self.data_raw_X), axis = 1 )
            self.data_raw_Dx = 2 * self.data_raw_X
        elif initialization_mode == 'zeros':
            self.data_raw_Y = np.zeros( (n_initial_data,1) )
            self.data_raw_Dx = np.zeros( (n_initial_data, self.nFeatures) )
        else:
            raise ValueError('convex nn initialization mode not supported')
            
        for i in range(50):
            self.train(False)
        
    def normalize_data(self, x, dx = None, y = None):
        dataX = (x - self.mean_x) / self.std_x
        
        if y is None:
            return dataX
        else:
            dataDx = dx  / self.std_y * self.std_x
            dataY = (y - self.mean_y) / self.std_y   
            dataY = dataY.reshape((-1, 1)).astype(np.float32)
            return dataX, dataDx, dataY

    def update_norm_params(self):
        self.std_x = np.std(self.data_raw_X, axis = 0)
        self.std_x = np.where(self.std_x < 10**-4, 1., self.std_x )
        self.mean_x = np.mean(self.data_raw_X, axis = 0)
        
        self.std_y = np.std(self.data_raw_Y, axis = 0)
        self.std_y = np.where(self.std_y < 10**-4, 1., self.std_y )
        self.mean_y = np.mean(self.data_raw_Y, axis = 0)

    def un_normalize_y(self, y):
        return y * self.std_y + self.mean_y

    def close(self):
        self.close_session()

    def close_session(self):
        self.sess.close()

    def get_param_nn(self):
        return [ param.eval(session=self.sess) for param in self.theta_] 
    
    def get_all_param_nn(self):
        return [ param.eval(session=self.sess) for param in self.all_var_] 

    def reset_wv(self):
        pass # no reset / will be used as "warm start"

    def copy(self, nn):
        p = nn.get_param_nn()
        self.copy_param(p)
        self.std_x = nn.std_x
        self.mean_x = nn.mean_x
        
        self.std_y = nn.std_y
        self.mean_y = nn.mean_y


    def copy_param(self, param):
        copy_tf = [tf.assign(self.theta_[i], p ) for i, p in enumerate(param) ]
        self.sess.run( copy_tf )
        
    def save(self, string):
        path = "wv_saved/concave_nn_" + string + ".dat"
        f = open( path, "w")
        pickle.dump(self.get_param_nn(),f)
        f.close()

    def load(self, string):
        path = "wv_saved/concave_nn_" + string + ".dat"
        f = open( path, "r")
        param = pickle.load(f)
        f.close()
        self.copy_param(param)

    def get_log(self):
        dict_log = {}
        dict_log['x'] = self.dataX
        dict_log['y'] = self.dataY
        x_dim = self.dataX.shape[1]
        if x_dim == 1:
            x_eval = np.linspace(-2,2,50).reshape((-1, 1)).astype(np.float32)
        elif x_dim == 2:
            x_eval = np.mgrid[-2:2:0.1, -2:2.:0.1].reshape(2,-1).T

        if x_dim <= 2 :
            dict_log['x_eval'] = x_eval
            dict_log['y_eval'] = self.sess.run(self.y_,feed_dict={self.x_: x_eval})
        return dict_log



class Convex_Relu_nn(Convex_NN):
    def __init__(self, next_state_variable, name = ""):
        super(Convex_Relu_nn, self).__init__(next_state_variable, name, fitting_mode = True )
        
        for i_layer, n_nodes in enumerate(self.n_hidden_sz):
            size_input = self.nFeatures
            if i_layer != 0 :
                size_input += self.n_hidden_sz_input[i_layer]
            self.w_nn.append( [ np.append(np.zeros(size_input), [- 1.]) for _ in range(n_nodes) ] )
        #self.rhs_norm_y = np.zeros(1)
        #self.coeff_norm_y = np.zeros(2)

        ##########################
        self.nn_var = self.get_nn_var()
        self.var_bound = self.make_variable_bound()
        self.non_print_var = self.nn_var


    def nn_design(self, x):
        max_norm = tf.contrib.keras.constraints.max_norm
        #PReLU = tf.contrib.keras.layers.PReLU(alpha_constraint = max_norm(1))
        dense = tf.contrib.layers.fully_connected
        ReLU = tf.nn.relu
        
        
        y = x
        z = []
        
        for i, n_hidden in enumerate(self.n_hidden_sz):
            new_layer = []
            
            new_layer.append( dense(
                y, n_hidden, activation_fn = None, 
                scope = 'z_y{}'.format(i) ) 
                )
            
            if z !=  []:
                # add positive matrix, no bias necessary
                new_layer.append( dense(
                    z[-1], n_hidden, activation_fn = None, 
                    scope = 'z_z{}_proj'.format(i),
                    biases_initializer = None ) 
                    )    

           
            # last layer : no activation fn
            if i != len(self.n_hidden_sz)-1 :
                z.append( ReLU(tf.add_n(new_layer)) )
            else:
                z.append( tf.add_n(new_layer) )
                   
        return z[-1]


    def get_constraints(self):
        constraint_list = []
        #self.alpha_prelu = []

        # self.rhs_norm_x = self.mean_x #np.zeros(self.nFeatures)
        self.rhs_norm_x[:] =  self.mean_x #/ self.std_x
        self.coeff_norm_x[:,0] = - self.std_x
        
        param_nn = self.get_param_nn()
        param_number = 0
        for i_layer, n_nodes in enumerate(self.n_hidden_sz):
            # unconstrained matrix
            layer_weight_matrix = param_nn[param_number]
            param_number += 1
            layer_bias = param_nn[param_number]
            param_number += 1
            if i_layer != len(self.n_hidden_sz)-1:
                pass
                #prelu_bias = param_nn[param_number]
                #self.alpha_prelu[i_layer] =  prelu_bias
                #param_number += 1
            # positive matrix
            if i_layer != 0 :
                # TODO : verify if extend
                layer_weight_matrix = np.append(layer_weight_matrix, param_nn[param_number], axis = 0 )
                param_number += 1
            
            for j_node in range(n_nodes): 
                self.w_nn[i_layer][j_node][:-1] = layer_weight_matrix[:,j_node]
                self.rhs_layer[i_layer][j_node] = - layer_bias[j_node]
            
       
        for i_input in range(self.nFeatures):
            constraint_list.append( 
                    linear_constraint( [ 'z_0_{}'.format(i_input) , self.next_state_variable[i_input] ], 
                                        self.coeff_norm_x[i_input], 'E', self.rhs_norm_x[i_input] ) )
        
            
        # constraints from network
        for i_layer, n_nodes in enumerate(self.n_hidden_sz):
            # w * z_i - z_i+1 > bias
            var_input = [ 'z_0_{}'.format(i_0) for i_0 in range(self.nFeatures)]
            if i_layer != 0 :
                var_input += [ 'z_{}_{}'.format(i_layer,i_input) for i_input in range(self.n_hidden_sz_input[i_layer])]
            
                
            for j_node in range(n_nodes):
                constraint_list.append( 
                    linear_constraint(  var_input + ['z_{}_{}'.format(i_layer+1,j_node)],
                                       self.w_nn[i_layer][j_node] , 'L', float(self.rhs_layer[i_layer][j_node]) ) )
                if i_layer != len(self.n_hidden_sz)-1 :
                    #constraint_list.append( 
                    #linear_constraint(  var_input + ['z_{}_{}'.format(i_layer+1,j_node)],
                    #                    np.append(self.alpha_prelu[i_layer][j_node] * self.w_nn[i_layer][j_node][:-1] , [ - 1. ] ), 'L', self.alpha_prelu[i_layer][j_node] * self.rhs_layer[i_layer][j_node] ) )
                    constraint_list.append( 
                        linear_constraint( ['z_{}_{}'.format(i_layer+1,j_node)], [ 1. ] , 'G', 0. ) )
            
        # constraint un-normalization
        constraint_list.append( 
                    linear_constraint( [ 'z_' + str(len(self.n_hidden_sz)) + '_0', 'water_value' ], 
                                       [float(self.std_y) , -1. ] , 'E', - self.mean_y  ) )
        
  
        return constraint_list



if __name__ ==  '__main__':
    
    dim  = 1

    state_var = Variable.list_variable_name_now[0:dim]
    
    convex_nn = Convex_Softmax_nn( state_var )
    convex_nn.update_standardized_value = True
    
    if dim == 1:
        convex_nn.data_raw_X = np.random.sample( (100, convex_nn.nFeatures) ) * 2. - 1.
        convex_nn.data_raw_Dx = 2 * convex_nn.data_raw_X
        convex_nn.data_raw_Y = np.sum( convex_nn.data_raw_X * convex_nn.data_raw_X, axis = 1 , keepdims = True  )
    elif dim == 2 :
        convex_nn.data_raw_X = np.mgrid[-4:4:0.5, -4:4.:0.5].reshape(2,-1).T
        convex_nn.data_raw_Dx = 2 * convex_nn.data_raw_X
        convex_nn.data_raw_Y = np.sum( convex_nn.data_raw_X * convex_nn.data_raw_X, axis = 1 , keepdims = True  )
        
    for i in range(100):
        convex_nn.train()
        # if i%100 ==0:
        #    convex_nn.plot_prediction()
    convex_nn.plot_prediction()
            
    x_val = [-1.] * dim #, -3]        
    wv_solver = convex_nn.evaluate_constraint( x_val )

    #print 'Solver - water_value = ', wv_solver
    #print 'Tensorflow - water_value = ', convex_nn.evaluate( np.array(x_val).reshape(-1,dim) )

    linear_nn = Convex_Softmax_nn(state_var, fitting_mode = False)
    linear_nn.copy(convex_nn)

    plt.scatter(linear_nn.data_raw_X, linear_nn.evaluate( linear_nn.data_raw_X), color = 'g')
    plt.show()
    #print 'Tensorflow linear softplus - water_value = ', linear_nn.evaluate( np.array(x_val).reshape(-1,dim) )
