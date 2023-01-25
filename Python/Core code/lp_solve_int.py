import numpy as np

#from past import autotranslate
#autotranslate(['lpsolve55'])
import lpsolve55
import lp_maker

class Lpsolve(object):
    #sens_val = ['L','E','G']
    #sens_tab = [-1, 0, 1]

    def __init__(self):
        self.n_col = 1
        self.n_row = 0
        self.lpsolve_prob = lpsolve55.lpsolve('make_lp', self.n_row, self.n_col)
        lpsolve55.lpsolve('set_verbose', self.lpsolve_prob, lpsolve55.CRITICAL)
        #lpsolve55.lpsolve('set_preferdual', self.lpsolve_prob, lpsolve55.SIMPLEX_DUAL_DUAL)
        lpsolve55.lpsolve('set_presolve', self.lpsolve_prob, lpsolve55.PRESOLVE_NONE)
        lpsolve55.lpsolve('is_piv_rule', self.lpsolve_prob, lpsolve55.PRICER_FIRSTINDEX )
        #self.lin_expr = []
        #self.sense = []
        self.rhs = []
        self.f = []
        self.a = []
        self.e = []
        self.vlb = []
        #self.obj_ind = []
        #self.obj_coeff = []

    def make_problem(self, lin_expr, sense, rhs, var_bounds, obj_var, obj_coeff):
        lpsolve55.lpsolve('delete_lp', self.lpsolve_prob)
        new_n_row = len(lin_expr)
        new_n_col = len(var_bounds)
        self.lpsolve_prob = lpsolve55.lpsolve('make_lp', self.n_row, self.n_col)
        lpsolve55.lpsolve('set_verbose', self.lpsolve_prob, lpsolve55.CRITICAL)
        #lpsolve55.lpsolve('set_preferdual', self.lpsolve_prob, lpsolve55.SIMPLEX_DUAL_DUAL)
        lpsolve55.lpsolve('set_presolve', self.lpsolve_prob, lpsolve55.PRESOLVE_NONE)
        lpsolve55.lpsolve('is_piv_rule', self.lpsolve_prob, lpsolve55.PRICER_FIRSTINDEX )
        # update matrix size
        self.update_matrix_size(new_n_row, new_n_col)
        
        self.n_row = new_n_row
        self.n_col = new_n_col
        
        self.f = self.make_obj_vector(obj_var, obj_coeff)
        self.a = self.make_constraint_mat(lin_expr)
        self.rhs = rhs
        self.e = self.make_sense_vec(sense)
        self.vlb = self.make_var_bounds(var_bounds)

        
            
        lpsolve55.lpsolve('set_mat', self.lpsolve_prob, self.a)
        lpsolve55.lpsolve('set_rh_vec', self.lpsolve_prob, self.rhs)
        lpsolve55.lpsolve('set_obj_fn', self.lpsolve_prob, self.f) 
        lpsolve55.lpsolve('set_minim', self.lpsolve_prob)  # set maximization
        #lpsolve55.lpsolve('set_verbose', self.lpsolve_prob, 5)

        for i in range(self.n_row):
            lpsolve55.lpsolve('set_constr_type', self.lpsolve_prob, i + 1, self.e[i])

        for i in range(self.n_col):
            lpsolve55.lpsolve('set_lowbo', self.lpsolve_prob, i + 1, self.vlb[i])

    def update_matrix_size(self,new_n_row, new_n_col):
        lpsolve55.lpsolve('resize_lp', self.lpsolve_prob, new_n_row, new_n_col)
        if self.n_row < new_n_row :
            for _ in range(self.n_row, new_n_row):
                n_col = min(new_n_col, self.n_col)
                if self.n_col > new_n_col:
                    pass
                lpsolve55.lpsolve('add_constraint', self.lpsolve_prob, np.zeros(n_col), 'LE', 0. )
        if self.n_col < new_n_col:
            for _ in range(self.n_col, new_n_col):
                lpsolve55.lpsolve('add_column', self.lpsolve_prob, np.zeros(new_n_row + 1) )

                 
    def make_obj_vector(self,obj_var, obj_coeff):
        obj_vec = np.zeros(self.n_col)
        obj_vec[obj_var] = obj_coeff
        return obj_vec

    def make_constraint_mat(self,lin_expr):
        mat_constraint = np.zeros([len(lin_expr),self.n_col])
        for i,(var, val), in enumerate(lin_expr):
            mat_constraint[i,var] = val
        return mat_constraint.tolist()

    def make_sense_vec(self,sense):
        return [s+'E' if s!='E' else 'EQ' for s in sense]

    def make_var_bounds(self, var_bounds):
        return [bound if bound != '-infinite' else - lp_maker.Infinite for bound in var_bounds]

    def solve(self):
        res = lpsolve55.lpsolve('solve', self.lpsolve_prob)
        if res == 0:
            return 0 # feasible problem
        else:
            return 1 # infeasible problem

    def print_problem(self):
        lpsolve55.lpsolve('set_outputfile', self.lpsolve_prob, 'test_lpsolve.txt')
        lpsolve55.lpsolve('print_lp', self.lpsolve_prob)

    def get_decisions(self):
        return lpsolve55.lpsolve('get_variables', self.lpsolve_prob)[0]

    def get_dual_values(self):
        return lpsolve55.lpsolve('get_dual_solution', self.lpsolve_prob)[0]

    def get_objective_value(self):
        return lpsolve55.lpsolve('get_objective', self.lpsolve_prob)

    def get_num_constraints(self):
        return lpsolve55.lpsolve('get_Nrows', self.lpsolve_prob)

    # def add_constraints(self,lin_exprs, senses, rhs):
    #     for i, (var, val) in enumerate(lin_exprs):
    #         lin_coeff = np.zeros(self.n_col)
    #         lin_coeff[var] = val
    #         lpsolve55.lpsolve('add_constraint', self.lpsolve_prob, lin_coeff,
    #                           senses[i]+'E' if senses[i]!='E' else 'EQ', rhs[i])

    # def set_objective_coefficient(self, var, coeff):
    #     self.f[var] = coeff
    #     lpsolve55.lpsolve('set_obj', self.lpsolve_prob, self.f)

    # def delete_constraints(self,range_constraints):
    #     # range constraint must be in descending order
    #     for row in sorted(range_constraints, reverse=True):
    #         lpsolve55.lpsolve('del_constraint', self.lpsolve_prob, 1 + row)

    # def set_linear_constraints_components(self,lin_expr_pair):
    #     # TODO : update matrix and test if faster
    #     for id_row, cons in lin_expr_pair:
    #         (id_col, val) = cons
    #         val_non_sparse = np.zeros(self.n_col, dtype = np.float32 )
    #         val_non_sparse[id_col] = val
    #         lpsolve55.lpsolve('set_rowex', self.lpsolve_prob, 1+id_row, val_non_sparse)

    # def set_linear_constraints_rhs(self,rhs_pair):
    #     for id, val in rhs_pair:
    #         lpsolve55.lpsolve('set_rh', self.lpsolve_prob, 1+id, val)

    # def declare_variable_names(self, variable_list):
    #     lpsolve55.lpsolve('set_col_name',self.lpsolve_prob,variable_list)
        
    # def get_coefficients_constraint(self, id_constraint, id_var):
    #     row_list = []
    #     for id_c in id_constraint:
    #         row_coeff = lpsolve55.lpsolve('get_row', self.lpsolve_prob, id_c + 1 )[0]
    #         row_list.append( [ row_coeff[id_v] for id_v in id_var] )
    #     return row_list
        
    # def get_rhs_constraint(self,id_constraint):
    #     return [ lpsolve55.lpsolve('get_rh', self.lpsolve_prob, id_c + 1) for id_c in id_constraint ]
        
