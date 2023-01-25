import pickle
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
import numpy as np

# initialization function: plot the background of each frame
def init():
    return lines
#    lines[0].set_data([], [])
#    lines[1].set_offsets(np.c_[ [], [] ])
#    lines[2].set_offsets(np.c_[ [], [] ])
    

#####  2D plot #############
def init_2D_plot():
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    plt.axes(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
    lines = [plt.plot([], [], lw=2)[0], 
             plt.scatter([], [], color='green', alpha = 0.4), 
             plt.scatter([], [], color ='red' , alpha = 0.4),
             plt.text(-2, 2, 'Time step : ', fontsize=15)
             ]
    return fig, [], lines

# animation function.  This is called sequentially
def animate_2D(i):
    is_n_iter = df['n_iter'] == frame_id[i]
    is_ts = df['ts'] == ts
    x_eval = df[is_n_iter & is_ts]['x_eval'].get_values()[0]
    y_eval = df[is_n_iter & is_ts]['y_eval'].get_values()[0]
    x = df[is_n_iter & is_ts]['x'].get_values()[0]
    y = df[is_n_iter & is_ts]['y'].get_values()[0]
    lines[0].set_data(x_eval, y_eval)
    lines[1].set_offsets(np.c_[ x[:50], y[:50] ])
    lines[2].set_offsets(np.c_[ x[50:], y[50:] ])
    lines[3].set_text('n_iter ' + str(frame_id[i]) )
    return lines

#####  3D plot #############
def init_3D_plot():
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    # ax.set_xlim3d(-4.5, 4.5)
    # ax.set_ylim3d(-4.5, 4.5)
    # ax.set_zlim3d(-4.5, 4.5)
    x_eval, y_eval, z_eval, x, y, z = get_data_3D(0)
#    plt.axes(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), zlim=(-2.5, 2.5))
    lines = [ax.plot_surface(x_eval, y_eval, z_eval, cmap = cm.coolwarm, linewidth=0, antialiased = False), 
             ax.scatter([], [], [], color='green', alpha = 0.4), 
             ax.scatter([], [], [], color ='red' , alpha = 0.4),
             ax.text2D(0, 0, 'Time step : ', transform = ax.transAxes, fontsize=15)
             ]
    return fig, ax, lines


def animate_3D(i):

    x_eval, y_eval, z_eval, x, y, z = get_data_3D(i)

    lines[0].remove()
    lines[0] = ax.plot_surface(x_eval, y_eval, z_eval,cmap = cm.coolwarm, linewidth=0, antialiased = False)
    lines[1]._offsets3d = ([ x[:50], y[:50], z[:50] ])
    lines[2]._offsets3d = ([ x[50:], y[50:], z[50:] ])
    lines[3].set_text('n_iter ' + str(frame_id[i]) )
    return lines
    
    
def get_data_3D(frame_number):
    is_n_iter = df['n_iter'] == frame_id[frame_number]
    is_ts = df['ts'] == ts
    xy_eval = df[is_n_iter & is_ts]['x_eval'].get_values()[0]
    z_eval = df[is_n_iter & is_ts]['y_eval'].get_values()[0]
    n_resize = int(sqrt(len(xy_eval)))
    x_eval = xy_eval[:,0].reshape((n_resize,n_resize))
    y_eval = xy_eval[:,1].reshape((n_resize,n_resize))
    z_eval = z_eval.reshape((n_resize,n_resize))

    xy = df[is_n_iter & is_ts]['x'].get_values()[0]
    x = xy[:,0]
    y = xy[:,1]
    z = np.squeeze(df[is_n_iter & is_ts]['y'].get_values()[0])
    
    return x_eval, y_eval, z_eval, x, y, z


anim_running = True

def onClick(event):
    global anim_running
    if anim_running:
        anim.event_source.stop()
        anim_running = False
    else:
        anim.event_source.start()
        anim_running = True

### main #####
file = open("log_cvx_nn.pickle",'rb')
log_dict_list = pickle.load(file)
file.close()
df = pd.DataFrame(log_dict_list)

x_dim = log_dict_list[0]['x'].shape[1]
if x_dim == 1 :
    mode = '2D'
elif x_dim == 2 :
    mode = '3D'
else:
    raise ValueError('Two much input dimensions')

ts = 1

frame_id = df.n_iter.unique()
n_frames = len(frame_id)

if mode == '2D':
    init_plot = init_2D_plot
    animate = animate_2D
else:
    init_plot = init_3D_plot
    animate = animate_3D
    
fig, ax, lines = init_plot()
anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=500)#, blit=True)
fig.canvas.mpl_connect('button_release_event', onClick)
# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#anim.save('cvx_nn_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()

