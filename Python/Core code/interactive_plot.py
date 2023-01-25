from os.path import join, dirname
import datetime
import numpy as np
import itertools  
import argparse

from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, DataRange1d, Select, Slider, RangeSlider, RadioGroup
from bokeh.palettes import Spectral11, viridis
from bokeh.plotting import figure
from bokeh.models import HoverTool

import time



def load_results_f(file):
    f = open( str(file) , 'r' )

    data = []

    l = f.readline()
    variables = l.replace('\t',' ').split(' ')

    l = f.readline()

    while l:
        data.append( np.array( l.replace('\t',' ').split(' '), dtype = np.float32 ) )
        l = f.readline()

    data = np.array(data)
    dict_data = dict()
    for i, var in enumerate(variables):
        dict_data[var] = data[:,i]

    return variables, dict_data


def get_dataset(data, key, id_start, id_end, horizon):
    time_serie = data[key]
    time_serie = np.reshape(time_serie, [-1,horizon])
    time_serie = time_serie[id_start:id_end,:]
    return time_serie

def make_plot(source, title):
    plot = figure( plot_width=1400, tools=[hover], toolbar_location = "below") #x_axis_type="datetime",
    plot.title.text = title

    plot.multi_line(xs = 'xs', ys = 'ys', line_color='line_color', line_width = 5, source = source ) # 

    #plot.x_range = DataRange1d(range_padding=0.0)
    #plot.y_range = DataRange1d(range_padding=5.0)

    # fixed attributes
    #plot.xaxis.axis_label = None
    #plot.yaxis.axis_label = "Temperature (F)"
    #plot.axis.axis_label_text_font_style = "bold"
    #plot.grid.grid_line_alpha = 0.3

    return plot

def update_plot(attrname, old, new):
    var = var_select.value
    id_start, id_end = year_select.value
    id_start = int(id_start) - 1
    id_end = int(id_end)
    plot.title.text = "Time serie for " + var

    new_time_serie = get_dataset(raw_data, var, id_start, id_end, horizon)
    type = type_select.active
    if type == 0:
        y = new_time_serie
    elif type == 1 :
        y = np.array([np.min(new_time_serie,axis=0), np.mean(new_time_serie,axis=0),
             np.max(new_time_serie,axis=0)])
    else:
        pass
    
    num_line = y.shape[0]

    my_palette = [Spectral11[i%11] for i in range(num_line)]
    source.data = dict(
        xs = [ range(1,horizon+1) for _ in range(num_line) ]  ,
        ys = y.tolist(),
        line_color = my_palette,
        )


parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default='simulation_log_new_CFDP_w_reopt.txt')
parser.add_argument('--horizon', type=int, default=12)
args = parser.parse_args()

horizon = args.horizon
file_name = args.file_name


variables, raw_data = load_results_f(file_name)
variable = variables[0]
len_data_set = len(raw_data[variables[0]]) / horizon

var_select = Select(value=variable, title='Variable', options=sorted(variables))
year_select = RangeSlider(title="Year selection", start=1, end=len_data_set, value=(1,len_data_set), step=1)
type_select = RadioGroup(labels=["Traces", "Statistics"], active=0)
#max_year = Slider(title="End Year released", start=1, end=len_data_set, value=len_data_set, step=1)


source = ColumnDataSource(data=dict(xs=[], ys=[], line_color=[]))

hover = HoverTool(tooltips=[
    ("(x,y)", "($x, $y)")])
#source = get_dataset(raw_data, variable)
plot = make_plot(source, "Time serie for " + variable)

var_select.on_change('value', update_plot)
#min_year.on_change('value', update_plot)
year_select.on_change('value', update_plot)
type_select.on_change('active', update_plot)
#distribution_select.on_change('value', update_plot)

controls = [var_select, year_select,type_select]
#for control in controls:
#    control.on_change('value', lambda attr, old, new: update_plot())

update_plot([], [], [])

control_layout = column(controls)
curdoc().add_root(column(control_layout, plot))
curdoc().title = file_name

