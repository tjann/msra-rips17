import collections
import sys
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ERROR_GRAPH_OUTPUT_PATH = '../../graphs'
TIME_GRAPH_OUTPUT_PATH = '../../graphs'

file_names = ['128.txt', '256.txt', '512.txt']
num_epoch = 160
num_worker = 4
num_trials = 3

def parse_output_file(file_name):
    with open(file_name, 'r') as f:
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
   

# modify this to accomodate 3 trials


# error_lists = [parse_output_file(i)[0] for i in file_names]
# time_lists = [parse_output_file(i)[1] for i in file_names]
error_lists = []
time_lists = []
for j in range(len(file_names)):
    temp_error = []
    temp_time = []
    for i in range(num_trials):
        temp_error.append(parse_output_file('res-' + str(i) + '-' +file_names[j])[0])
        temp_time.append(parse_output_file('res-' + str(i) + '-' +file_names[j])[1])
    error_lists.append(np.average(temp_error, axis=0))
    time_lists.append(np.average(temp_time, axis=0))

   

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
ax.legend(loc='upper right', fontsize=18)
# plt.title('Training Error of CNN on CIFAR-10 data (BMUF) - Async')
plt.xlabel("Number of epoch", fontsize=18)
plt.ylabel("Training error in %", fontsize=18)
plt.savefig(ERROR_GRAPH_OUTPUT_PATH + '/async/cnn-async-error.pdf')

# PLOT TIME GRAPH
fig, ax = plt.subplots()
for i in range(len(file_names)):
    ax.plot(list(range(1, len(time_lists[i][1:]) + 1)), time_lists[i][1:], color=color_list[i], label=label_list[i])
ax.legend(loc='center right', fontsize=14)
# ADD AVERAGE TIME
avg_text = ''
for i in range(len(file_names)):
    avg_text += avg_list[i] + ': ' +str(np.average(time_lists[i])) + '\n'
plt.text(0.0, 0.5, avg_text,
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes,
     fontsize=12)
# plt.title('Time per epoch of CNN on CIFAR-10 data (BMUF) - Async')
plt.xlabel("Number of epoch", fontsize=18)
plt.ylabel("Time in second", fontsize=18)
plt.savefig(TIME_GRAPH_OUTPUT_PATH + '/async/cnn-async-time.pdf')

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
print(error_lists[1])

for i in range(len(file_names)):
    ax.plot(accum_time[i], error_lists[i], color=color_list[i], label=label_list[i])

ax.legend(loc='upper right', fontsize=18)
# plt.title('Training Error by Time of CNN on CIFAR-10 data (BMUF) - Async')
plt.xlabel("Time", fontsize=18)
plt.ylabel("Training error in %", fontsize=18)
plt.savefig(ERROR_GRAPH_OUTPUT_PATH + '/async/cnn-async-error-by-time.pdf')

