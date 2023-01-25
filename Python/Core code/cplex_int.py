import cplex
from cplex.exceptions import CplexError

class Cplex(object):
    def __init__(self):
        self.cplex_prob = self.initialize_cplex()

    def initialize_cplex(self):
        cplex_prob = cplex.Cplex()
        cplex_prob.set_log_stream(None)
        cplex_prob.set_error_stream(None)
        cplex_prob.set_warning_stream(None)
        cplex_prob.set_results_stream(None)
        cplex_prob.objective.set_sense(cplex_prob.objective.sense.minimize)

        cplex_prob.set_problem_type(cplex_prob.problem_type.LP)
        cplex_prob.parameters.lpmethod.set(cplex_prob.parameters.lpmethod.values.dual) # Dual
        cplex_prob.parameters.preprocessing.presolve.set(cplex_prob.parameters.preprocessing.presolve.values.off)
        return cplex_prob

    def make_problem(self, lin_expr, sense, rhs, lower_bounds, obj_var, obj_coeff):
        self.cplex_prob.end()
        self.cplex_prob = self.initialize_cplex() #TODO : just reset
        self.lin_expr = lin_expr
        self.sense = sense
        self.rhs = rhs
        self.obj_var = obj_var
        self.obj_coeff = obj_coeff

        self.declare_variable(lower_bounds)
        self.add_constraints(self.lin_expr, self.sense, self.rhs)
        self.set_objective_coefficient(self.obj_var, self.obj_coeff)

    def solve(self):
        self.cplex_prob.solve()
        if self.cplex_prob.solution.get_status() > 1: # unfeasible problem
            return True
        else:
            return False

    def print_problem(self):
        self.cplex_prob.write('test_cplex.lp')

    def get_decisions(self):
        return self.cplex_prob.solution.get_values()

    def get_dual_values(self):
        return self.cplex_prob.solution.get_dual_values()

    def get_objective_value(self):
        return self.cplex_prob.solution.get_objective_value()

    def get_num_constraints(self):
        return self.cplex_prob.linear_constraints.get_num()

    def delete_constraints(self,range_constraints):
        self.cplex_prob.linear_constraints.delete(range_constraints)

    def set_linear_constraints_components(self,lin_expr_pair):
        self.cplex_prob.linear_constraints.set_linear_components(lin_expr_pair)

    def set_linear_constraints_rhs(self,rhs_pair):
        self.cplex_prob.linear_constraints.set_rhs(rhs_pair)

    def declare_variable(self, var_bound):
        self.cplex_prob.variables.add( obj = [0. for _ in range(len(var_bound))])
        for i, value in enumerate(var_bound):
            if value == '-infinite':
                self.cplex_prob.variables.set_lower_bounds(i, -cplex.infinity )
            # else:
            #    var>= 0

    def set_objective_coefficient(self, variables, coefficients):
        # update objective
        self.cplex_prob.objective.set_linear(zip(variables, coefficients))

    # method to add constraints in batch
    def add_constraints(self, lin_expr, senses, rhs):
        # add constraint to cplex problem
        self.cplex_prob.linear_constraints.add(lin_expr=lin_expr, senses=senses, rhs=rhs)

    def declare_variable_names(self, variable_list):
        self.cplex_prob.variables.set_names(zip(range(len(variable_list)),variable_list))
