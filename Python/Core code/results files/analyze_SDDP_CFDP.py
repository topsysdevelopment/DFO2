import numpy as np
import matplotlib.pyplot as plt


def read_file(path):
	myList = []
	f = open(path, "r") #opens file with name of "test.txt"
	for line in f:
	    myList.append(line.split())
	return np.array(myList)


if __name__ == '__main__':
	sddp_file = ['res_SDDP.txt','res_SDDP_2.txt']
	for file in sddp_file:
		res_SDDP = read_file(file)
		print(res_SDDP)
		x_sddp = [(10*(9.*11.+1.+12))*x for x in range(1,len(res_SDDP)+1)]
		plt.plot(x_sddp, res_SDDP, alpha=0.5, label ='SDDP', linewidth = 3., color = 'blue')
	
	cfdp_file = ['res_CFDP.txt','res_CFDP_2.txt']
	for file in cfdp_file:
		res_CFDP = read_file(file)
		x_cfdp = [100*(12)*x for x in range(1,len(res_CFDP)+1)]
		plt.plot(x_cfdp, res_CFDP, alpha=0.5, label ='CFDP', linewidth = 3., color = 'green')

	axes = plt.gca()
	axes.set_ylim([-250,-200])
	axes.legend()
	plt.show()
