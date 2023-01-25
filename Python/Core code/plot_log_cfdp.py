import pickle
import pandas as pd 
import matplotlib.pyplot as plt

#with open('log_cvx_nn.pickle', 'rb') as handle:

file = open("log_cvx_nn.pickle",'rb')
log_dict_list = pickle.load(file)
file.close()

df = pd.DataFrame(log_dict_list)

ts = 6
for n_iter in df.n_iter.unique():
    is_n_iter = df['n_iter'] == n_iter
    is_ts = df['ts'] == ts
    x_eval = df[is_n_iter & is_ts]['x_eval'].get_values()[0]
    y_eval = df[is_n_iter & is_ts]['y_eval'].get_values()[0]
    x = df[is_n_iter & is_ts]['x'].get_values()[0]
    y = df[is_n_iter & is_ts]['y'].get_values()[0]
    plt.plot(x_eval,y_eval)
    plt.scatter(x[:50],y[:50], color='green')
    plt.scatter(x[50:],y[50:], color='red')
    axes = plt.gca()
    axes.set_xlim([-3.,3.])
    axes.set_ylim([-3.,3.])
    plt.show()
