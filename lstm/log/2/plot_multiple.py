import collections
import sys
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

GRAPH_OUTPUT_PATH = '../../graphs'

file_names = ['res-240.txt', 'res-480.txt', 'res-720.txt']
num_epoch = 50
num_worker = 3

def parse_output_file(file_name):
    with open('../../log/2/' + file_name, 'r') as f:
        list_of_epoch_information = [l for l in f]
        list_of_epoch_information = [l for l in list_of_epoch_information if re.match(r'Finished Epoch', l)]
    i = 0
    error_epoch = collections.OrderedDict()
    time_epoch = collections.OrderedDict()
    while i < len(list_of_epoch_information):
        temp = list_of_epoch_information[i].split(':')
        temp_string = ''
        for t in temp[1:]:
            temp_string += t
        try:
            error_epoch[temp[0]] = float(re.search(r'(\d|\.)+\%', temp_string.strip()).group().split('%')[0])
        except:
            pass
        try:
            time_epoch[temp[0]] = float(re.search(r'(\d|\.)+s', temp_string.strip()).group().split('s')[0])
        except:
            pass
        i += 1
    error_list = []
    time_list = []
    for j in error_epoch:
        error_list.append(error_epoch[j])
        time_list.append(time_epoch[j])
    return (error_list, time_list)
   
error_lists = [parse_output_file(i)[0] for i in file_names]
time_lists = [parse_output_file(i)[1] for i in file_names]
color_list = ['red', 'green', 'blue', 'magenta', 'black']
label_list = ['Sync period: 240', 'Sync period: 480', 'Sync period: 720']
avg_list = ['Avg 240', 'Avg 480', 'Avg 720']
# Check if there is any funky value in the list
for i in range(len(error_lists)):
    for j in range(len(error_lists[i])):
        if error_lists[i][j] > 100:
            print('ERROR i: ', i, 'j: ', j, 'value: ', error_lists[i][j])
# PLOT ERROR GRAPH
fix, ax = plt.subplots()
for i in range(len(file_names)):
    ax.plot(list(range(1, len(error_lists[i]) + 1)), error_lists[i], color=color_list[i], label=label_list[i])
ax.legend(loc='upper right')
plt.title('Training Error of LSTM on NMT')
plt.xlabel("Number of epoch")
plt.ylabel("Training error in %")
plt.savefig(GRAPH_OUTPUT_PATH + '/2/error-240,480,720.pdf')

# PLOT TIME GRAPH
fig, ax = plt.subplots()
for i in range(len(file_names)):
    ax.plot(list(range(1, len(time_lists[i][1:]) + 1)), time_lists[i][1:], color=color_list[i], label=label_list[i])
ax.legend(loc='upper right')
# ADD AVERAGE TIME
avg_text = ''
for i in range(len(file_names)):
    avg_text += avg_list[i] + ': ' +str(np.average(time_lists[i])) + '\n'
plt.text(0.2, 0.9, avg_text,
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.title('Time per epoch of LSTM on NMT')
plt.xlabel("Number of epoch")
plt.ylabel("Time in second")
plt.savefig(GRAPH_OUTPUT_PATH + '/2/time-240,480,720.pdf')

# PLOT TIME x Error Graph
fig, ax = plt.subplots()
accum_time = []
for i in range(len(file_names)):
    accum_time.append([0])

for j in range(len(file_names)):
    for i in time_lists[j]:
        accum_time[j].append(float(i) + accum_time[j][-1])

for i in range(len(file_names)):
    accum_time[i] = accum_time[i][1:]

for i in range(len(file_names)):
    ax.plot(accum_time[i], error_lists[i], color=color_list[i], label=label_list[i])

ax.legend(loc='upper right')
plt.title('Training Error by Time of LSTM for NMT Data')
plt.xlabel("Time")
plt.ylabel("Training error in %")
plt.savefig(GRAPH_OUTPUT_PATH + '/2/error-by-time-240,480,720.pdf')
