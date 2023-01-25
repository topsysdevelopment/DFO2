import sys
import numpy as np
import matplotlib.pyplot as plt


def load_results_f(file):
	f = open( str(file) , 'r' )

	data = []

	l = f.readline()
	l = f.readline()

	while l:
		data.append( np.array( l.split( '\t' ) ) )
		l = f.readline()

	return np.array(data)

def plot_time_serie(data, show_plot = True, color = 'b' ):
	plt.plot(data.reshape( 12 , -1), color = color )
	if show_plot :
		plt.show()
		
		
if __name__ == '__main__' :
	file = sys.argv[1]
	load_results(file)