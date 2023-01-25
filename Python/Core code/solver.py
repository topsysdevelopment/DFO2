import copy
import numpy as np
import sys

def returnList(vec):
    if isinstance(vec, list):
        return vec
    elif type(vec).__module__ == np.__name__:
        if np.shape(vec):
            return vec.tolist()
        else:  # if float
            return [vec.tolist()]
    elif isinstance(vec, float):
        return [vec]
    elif isinstance(vec, int):
        return [float(vec)]
    else:
        print('format not supported in returnList')
        sys.exit()
    
        
class Linear_problem(object):
    """ Linear Problem class.
    """
    def __init__(self, solver):
        """ 
        Args: 
            solver (string) : solver choice. Must be Lp_solve or Cplex
        """
        self.decisions = [ ]
        self.dual_values = [ ]
        self.state_variable_constraints_idx = {}
        self.objectiveValue = 0.
        
        if solver == 'Cplex':
            from cplex_int import Cplex
            self.prob = Cplex()
        elif solver == 'Lpsolve' or solver == 'Lp_solve':
            from lp_solve_int import Lpsolve
            self.prob = Lpsolve()
        else:
            raise 'Solver not managed'

    def populate_problem(self, objectives, constraints, variable_bound):
        """ Function to create the problem in the solver interface. 
            First, the list of all variables is made, providing a mapping between each individual variables names and index of the matrix formulation.
            Then the constraint matrix, the objective vector and the lower bound are written in a matrix format.
            Finally the problem is ready to be solved.
            Note: No upper bound setting function has been implemented. 
            
        Args:
            objectives (list of objective): list of objective object
            constraints (list of constraints): list of constraint object
            variable_bound (dictionnary): dictionnary of (variable, lower bound value)
        """

        self.variable_list, self.variable_to_indice_dict = self.make_variable_list(objectives, constraints)

        if constraints:
            cons_lin_expr, cons_senses, cons_rhs = self.make_constraints(constraints)        
        if objectives:
            objective_coefficients, objective_variables = self.make_objective(objectives)
        if variable_bound:
            var_bounds = self.make_lower_bound(variable_bound)

        self.prob.make_problem(cons_lin_expr, cons_senses, cons_rhs, var_bounds,
                               objective_variables, objective_coefficients)

    def make_variable_list(self, objectives, constraints ):
        """ Extract all the the variable name fron the model.

        Args:
            objectives (list of objective): list of objective object
            constraints (list of constraints): list of constraint object
        
        Returns:
            variable_list (list of strings): list of all the variable names ordered in alphabetical order
            variable_to_indice_dict : dictionnary (variable name, index of variable)

        """

        variable_list = []
        
        for c in constraints:
            variable_list.extend(c.variable_name)
        for o in objectives:
            variable_list.extend(c.variable_name)

        variable_list = sorted(list(set(variable_list)))
        variable_to_indice_dict = {variable_list[i]: i for i in range(len(variable_list))}

        return variable_list, variable_to_indice_dict

    def make_constraints(self, constraints):
        """ Write the constraints in a matrix form.

        Args:
            constraints (list of constraints): list of constraint object
        
        Returns:
            lin_expr (list of list) : list of [constraint variable index (list) , constraint coefficient (list) ]
            senses (list): list of constraint sense
            rhs (list): list of constraints right hand side value
        """

        lin_expr = []
        senses = []
        rhs = []

        for constraint in constraints:
            beggining_length = len(lin_expr)
            if isinstance(constraint, linear_constraint):
                lin_expr.extend(
                        [[self.get_variable_to_indice_dict(constraint.variable_name), constraint.coeff]])

                senses.extend(constraint.sense)
                rhsFormated = returnList(constraint.rightHandSide)
                rhs.extend(rhsFormated)

                # is state variable ?
                if constraint.stateVariable:
                    self.state_variable_constraints_idx[constraint.stateVariable] = len(senses) - 1

            elif isinstance(constraint, concave_piecewise_linear_contraint):
                ind_linar_expr = []
                for i, name_linar_expr in enumerate(constraint.linear_expr):
                    ind_linar_expr.extend(
                        [[self.get_variable_to_indice_dict(name_linar_expr[0]), name_linar_expr[1]]])
                lin_expr.extend(ind_linar_expr)
                senses.extend(constraint.sens)
                rhs.extend(constraint.rhs)
            end_length = len(lin_expr)
            # set constraint id
            # constraint.idx = range(beggining_length, end_length)

        return lin_expr, senses, rhs

    def make_objective(self, objectives):
        """ Write the objective in a matrix form.

        Args:
            objectives (list of objective): list of objective object
        
        Returns:
            obj_var (list of list) : list of objective variable index.
            obj_coeff (list): list of objective coefficient
        """

        # change coefficient if maximization objective
        # TODO min or max
        obj_var = []
        obj_coeff = []
        for objective in objectives:
            obj_var.extend(objective.coefficient)
            obj_coeff.extend([self.variable_to_indice_dict[x] for x in objective.variable])

        return obj_var, obj_coeff


    def make_lower_bound(self, var_bound):
        """ Write the lower bound in a matrix form. Value should be set to 'free' to indicate an infinte lower bound.

        Args:
            var_bound (dictionnary): dictionnaray of lower variable bounds
        
        Returns:
            lower_bound (list) : lower bound of variable in a matrix form.        
        """

        lower_bound = [0 for _ in self.variable_list]
        for (var, value) in var_bound.items():
            if var in self.variable_to_indice_dict:
                if value == 'free':
                    lower_bound[ self.variable_to_indice_dict[var]] = '-infinite'
        return lower_bound


    def get_variable_to_indice_dict(self, var):
        """ Return the indice of a variable ased on its string name.

        Args:
            var (list or string): list of variable string names or a variable string.
        
        Returns:
            list of int if var is a list or int if var is a string.        
        """

        if isinstance(var, list):
            return [self.variable_to_indice_dict[v] for v in var]
        else:
            return self.variable_to_indice_dict[var]


    def solve(self):
        """ This solves the problem. Must be called after populate_problem
        
        Returns:
            1 if the problem is unfeasible, 0 otherwise.
        """

        unfeasible_problem = self.prob.solve()

        self.decisions = self.prob.get_decisions()
        self.dual_values = self.prob.get_dual_values()
        self.objectiveValue = self.prob.get_objective_value()
        
        return unfeasible_problem


    def get_decisions(self, vars = []):
        """ This returns the decision values.

        Args : 
            vars (list): list of decision variable to return the value

        Returns :
            return decions for all variable in vars. 
            return all variable value if vars is empty.

        """

        if vars == []:
            return self.decisions
        else:
            return [ self.decisions[self.variable_to_indice_dict[var]] for var in vars]


    def get_dual_values(self, vars):
        """ This returns the dual values

        Args: 
            vars (list): list of decision variable to return the dual value

        Returns:
            return dual value for all variable in vars. 

        """
        return [self.dual_values[self.state_variable_constraints_idx[var]] for var in vars ]
                
      
class objective():
    """ Objectve class """
    def __init__(self, coefficient, variable, sense = 'min'):
        """ 
        Args:
            coefficient (list): coefficient associated to the variables
            variable (list) : variable names

        Kwargs: 
            sense : 'min' or 'max' 
        """
        assert len(coefficient) == len(variable)
        self.coefficient = coefficient
        self.variable = variable
        self.sense = sense


    def __str__(self):
        if len(self.coefficient) != len(self.variable):
        #if len(self.coeff.shape) == 2:
            coeff = self.coefficient[0]
        else:
            coeff = self.coefficient
        string = str(coeff[0]) + ' * '+ self.variable[0] 
        for i in range(1,len(self.variable)):
            string += ' + '
            string += str(coeff[i]) + ' * ' + self.variable[i] 
        return string

class linear_constraint():
    """ Linear constrain class. """
    dict_sense = { 'E' : '=' , 'G' : '>=' ,'L' : '<=' }

    def __init__(self, variable, coeff, sense, rightHandSide, state_variable = [] ):
        """ 
        Args:
            variable (list) : variable names
            coefficient (list): coefficient associated to the variables
			sense
			rightHandSide

        Kwargs: 
            state_variable : 'min' or 'max' 
        """

        assert len(variable) == len(coeff)
        self.variable_name = variable
        self.variable_id = []
        self.coeff = coeff
        self.sense = sense
        self.rightHandSide = rightHandSide
        self.stateVariable = state_variable

    def __str__(self):
        """ Representation of a constraint
        
        ex: print(linear_constraint( ['var1'], [2] , 'G' , 4. )) outputs
        2 * 'var1' >= 4.
        
        Returns:
            string representing the constraint
        """
        
        if len(self.coeff) != len(self.variable_name):
        #if len(self.coeff.shape) == 2:
            coeff = self.coeff[0]
        else:
            coeff = self.coeff
        string = str(coeff[0]) + ' * '+ self.variable_name[0] 
        for i in range(1,len(self.variable_name)):
            string += ' + '
            string += str(coeff[i]) + ' * '+ self.variable_name[i] 
        string += ' ' + self.dict_sense[self.sense] 
        string += ' ' + str(self.rightHandSide)
        return string

class piecewise_linear_contraint(object):
    def __init__(self, **kwargs):
        self.vary = kwargs['vary']
        self.varx = kwargs['varx']
        self.variable_name = [self.varx, self.vary]
        self.variable_id = []
        self.id = []
        self.nparts = kwargs['nparts'].tolist()
        if isinstance(self.nparts, list):
            self.nparts = int(self.nparts[0])
        self.bkpointx = kwargs['bkpointx'].tolist()
        if 'slope' in kwargs:
            """args = varx,vary,nparts,bkpointx,slope """
            self.bkpointx = self.bkpointx[:self.nparts - 1]
            self.slope = kwargs['slope'][:self.nparts].tolist()
            if 'intercept' in kwargs:
                self.intercept = kwargs['intercept']
            else:
                self.intercept = 0.
        elif 'bkpointy' in kwargs:
            self.slope =[]
            self.bkpointy = kwargs['bkpointy'].tolist()
            self.nparts -= 1


    def __str__(self):
        string =  "PWL constraint with variables: " + str(self.variable_name)
        return string

class concave_piecewise_linear_contraint(piecewise_linear_contraint):
    def __init__(self, **kwargs):
        super(concave_piecewise_linear_contraint, self).__init__( **kwargs)
        self.linear_expr =[]
        self.rhs = []
        self.sens = ['L'] * self.nparts

        if self.slope != []:
            y = [0.]
            x = self.bkpointx[0]
            for i in range(self.nparts):
                self.linear_expr.append([self.variable_name , [- self.slope[i], 1.]])
                self.rhs.append( y[-1] - self.slope[i] * x )
                if i < self.nparts-1:
                    y.append( y[-1] + self.slope[i] * (self.bkpointx[i] - x ) )
                    x = copy.copy(self.bkpointx[i])
            self.rhs = np.array(self.rhs)
            if 0. in self.bkpointx:
                idzero = self.bkpointx.index(0.)
                delta = self.intercept - y[idzero+1]
                self.rhs += delta
            elif self.bkpointx[0] > 0:
                delta = self.intercept - self.rhs[0]
                self.rhs += delta
            elif self.bkpointx[0] < 0:
                raise('error')
            self.rhs = self.rhs.tolist()

        elif self.bkpointy != []:
            for i in range(len(self.bkpointy)-1):
                slope = (self.bkpointy[i + 1] - self.bkpointy[i]) / (self.bkpointx[i + 1] - self.bkpointx[i])
                intercept = self.bkpointy[i] - slope * self.bkpointx[i]
                self.linear_expr.append([self.variable_name, [- slope, 1.]])
                self.rhs.append(intercept)

    def __str__(self):
        string = ''
        for i in range(self.nparts):
            if string != '':
                string+= '\n'
            string += str(- self.slope[i]) + '*' + self.variable_name[0] + ' + ' + self.variable_name[1]
            string += ' <= ' + str(self.rhs[i]) 
        return string