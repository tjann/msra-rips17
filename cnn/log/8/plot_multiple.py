import collections
import sys
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ERROR_GRAPH_OUTPUT_PATH = '../../error_graphs'
TIME_GRAPH_OUTPUT_PATH = '../../time_graphs'

file_names = ['res-128.txt', 'res-256.txt', 'res-512.txt']
num_epoch = 160
num_worker = 4

def parse_output_file(file_name):
    with open('../../log/8/' + file_name, 'r') as f:
        list_of_epoch_information = [l for l in f]
        list_of_epoch_information = [l for l in list_of_epoch_information if re.match(r'Finished Epoch', l)]
    print(len(list_of_epoch_information))
    count = 0
    error = []
    time = []
    for i in range(int(len(list_of_epoch_information) / num_worker)):
        temp_error = []
        temp_time = []
        for j in range(num_worker):
            curr_string = list_of_epoch_information[count + j]
            temp_error.append(float(re.search(r'(\d|\.)+\%', curr_string.strip()).group().split('%')[0]))
            temp_time.append(float(re.search(r'(\d|\.)+s', curr_string.strip()).group().split('s')[0]))
        error.append(np.average(temp_error))
        time.append(np.average(temp_time))
        count += num_worker
    return error, time
    
error_lists = [parse_output_file(i)[0] for i in file_names]
time_lists = [parse_output_file(i)[1] for i in file_names]
color_list = ['red', 'green', 'blue', 'magenta', 'black']
label_list = ['Sync period: 128', 'Sync period: 256', 'Sync period: 512']
avg_list = ['Avg 128', 'Avg 256', 'Avg 512']
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
plt.title('Training Error of CNN on CIFAR-10 data (Async)')
plt.xlabel("Number of epoch")
plt.ylabel("Training error in %")
plt.savefig(ERROR_GRAPH_OUTPUT_PATH + '/multiples/8/error-128,256,512.pdf')

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
plt.title('Time per epoch of CNN on CIFAR-10 data (Async)')
plt.xlabel("Number of epoch")
plt.ylabel("Time in second")
plt.savefig(TIME_GRAPH_OUTPUT_PATH + '/multiples/8/time-128,256,512.pdf')

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
plt.title('Training Error by Time of CNN on CIFAR-10 data (Async)')
plt.xlabel("Time")
plt.ylabel("Training error in %")
plt.savefig(ERROR_GRAPH_OUTPUT_PATH + '/multiples/8/error-by-time-128,256,512.pdf')
