import copy
import numpy as np

import utils
from __builtin__ import list

class Linear_problem(object):
    def __init__(self, solver):
        
        self.solver = solver
        
        self.decisions = [ ] #0. for _ in self.variable_list]
        self.dual_values = [ ] #0. for _ in self.variable_list]
        self.state_variable_constraints_idx = {}

        if solver == 'Cplex':
            from cplex_int import Cplex
            self.prob = Cplex()
        elif solver == 'Lpsolve' or solver == 'Lp_solve':
            from lp_solve_int import Lpsolve
            self.prob = Lpsolve()
        else:
            'Solver not managed'

    def populate_problem(self, objectives, constraints, variable_bound):
        self.variable_list, self.variable_to_indice_dict = self.make_variable_list(objectives, constraints)

        if constraints:
            cons_lin_expr, cons_senses, cons_rhs = self.make_constraints(constraints)        
        if objectives:
            objective_coefficients, objective_variables = self.make_objective(objectives)
        if variable_bound:
            var_bounds = self.make_lower_bound(variable_bound)

        self.prob.make_problem(cons_lin_expr, cons_senses, cons_rhs, var_bounds,
                               objective_variables, objective_coefficients)

    # Method to declare variables
    def make_variable_list(self, objectives, constraints ):
        variable_list = []

        for c in constraints:
            for variable in c.variable_name:
                if isinstance(variable, basestring):
                    variable_list.append(variable)
                else:
                    variable_list.extend(variable)
        for o in objectives:
            for variable in o.variable:
                if isinstance(variable, basestring):
                    variable_list.append(variable)
                else:
                    variable_list.extend(variable)

        variable_list = sorted(list(set(variable_list)))
        variable_to_indice_dict = {variable_list[i]: i for i in range(len(variable_list))}

        return variable_list, variable_to_indice_dict

    def make_constraints(self, constraints):
        lin_expr = []
        senses = []
        rhs = []

        for constraint in constraints:
            beggining_length = len(lin_expr)
            if isinstance(constraint, linear_constraint):
                if isinstance(constraint.variable_name[0], basestring):
                    lin_expr.extend(
                        [[self.get_variable_to_indice_dict(constraint.variable_name), constraint.coeff]])
                else:
                    lin_expr.extend(
                        [[self.get_variable_to_indice_dict(constraint.variable_name[0]), constraint.coeff[0]]])
                senses.extend(constraint.sense)
                rhsFormated = utils.returnList(constraint.rightHandSide)
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
            constraint.idx = range(beggining_length, end_length)

        return lin_expr, senses, rhs

    def make_objective(self, objectives):
        # change coefficient if minimization objective
        # TODO min or max
        obj_var = []
        obj_coeff = []
        for objective in objectives:
            obj_var.extend(objective.coefficient)
            obj_coeff.extend([self.variable_to_indice_dict[x] for x in objective.variable])

        return obj_var, obj_coeff


    def make_lower_bound(self, var_bound):
        lower_bound = [0 for _ in self.variable_list]
        for (var, value) in var_bound.iteritems():
            if self.variable_to_indice_dict.has_key(var):
                if value == 'free':
                    lower_bound[ self.variable_to_indice_dict[var]] = '-infinite'
        return lower_bound


    def get_variable_to_indice_dict(self, var):
        if isinstance(var, list):
            return [self.variable_to_indice_dict[v] for v in var]
        else:
            return self.variable_to_indice_dict[var]

    # solve method
    def solve(self):
        unfeasible_problem = self.prob.solve()

        self.decisions = self.prob.get_decisions()
        self.dual_values = self.prob.get_dual_values()
        
        self.objectiveValue = self.prob.get_objective_value()
        
        return unfeasible_problem


    def get_decisions(self, vars = []):
        if vars == []:
            return self.decisions
        else:
            return [ self.decisions[self.variable_to_indice_dict[var]] for var in vars]

    def get_dual_values(self, vars):
        return [self.dual_values[self.state_variable_constraints_idx[var]] for var in vars ]
        
       
class objective():
    def __init__(self, coefficient, name, sense = 'max'):
        self.coefficient = coefficient
        self.variable = name
        self.sense = sense


class linear_constraint():
    dict_sense = { 'E' : '=' , 'G' : '>=' ,'L' : '<=' }


    def __init__(self, variable, coeff, sense, rightHandSide, **kwargs ):
        # TODO : check input format
        self.variable_name = variable
        self.variable_id = []
        self.coeff = coeff
        self.sense = sense
        self.rightHandSide = rightHandSide
        if 'state_variable' in kwargs:
            self.stateVariable = kwargs['state_variable']
        else:
            self.stateVariable = []
        if 'is_dynamic' in kwargs:
            self.is_dynamic = kwargs['is_dynamic']
        else:
            self.is_dynamic = {}
        self.idx = []

    def __str__(self):
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
        string += ' ' + str(self.rightHandSide[0])
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

        if 'is_dynamic' in kwargs:
            self.is_dynamic = kwargs['is_dynamic']
        else:
            self.is_dynamic = {}

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
                pass
            self.rhs = self.rhs.tolist()

        elif self.bkpointy != []:
            for i in range(len(self.bkpointy)-1):
                slope = (self.bkpointy[i + 1] - self.bkpointy[i]) / (self.bkpointx[i + 1] - self.bkpointx[i])
                intercept = self.bkpointy[i] - slope * self.bkpointx[i]
                self.linear_expr.append([self.variable_name, [- slope, 1.]])
                self.rhs.append(intercept)
