import load_results as lr

var, res_SDDP = lr.load_results_f( 'simul_SDDP.txt' )
var, res_CFDP = lr.load_results_f( 'simul_CFDP.txt' )

lr.plot_time_serie(res_SDDP[:,0], False, 'b')
lr.plot_time_serie(res_CFDP[:,0], True, 'r')