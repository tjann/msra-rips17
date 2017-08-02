import collections
import sys
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ERROR_GRAPH_OUTPUT_PATH = 'error_graphs'
TIME_GRAPH_OUTPUT_PATH = 'time_graphs'

file_names = ['res-128.txt', 'res-1000.txt','res-1280.txt', 'res-2000.txt', 'res-12800.txt']

def parse_output_file(file_name):
    with open('log/7/' + file_name, 'r') as f:
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

# Check if there is any funky value in the list
for i in range(len(error_lists)):
    for j in range(len(error_lists[i])):
        if error_lists[i][j] > 100:
            print('ERROR i: ', i, 'j: ', j, 'value: ', error_lists[i][j])
'''
# PLOT ERROR GRAPH
fig, ax = plt.subplots()
ax.plot(list(range(1, len(error_lists[0]) + 1)), error_lists[0], color='red', label='Sync period:128')
ax.plot(list(range(1, len(error_lists[1]) + 1)), error_lists[1], color='black', label='Sync period:1000')
ax.plot(list(range(1, len(error_lists[2]) + 1)), error_lists[2], color='green', label='Sync period:1280')
ax.plot(list(range(1, len(error_lists[3]) + 1)), error_lists[3], color='magenta', label='Sync period:2000')
ax.plot(list(range(1, len(error_lists[4]) + 1)), error_lists[4], color='blue', label='Sync period:12800')
ax.legend(loc='upper right')
plt.title('Training Error of CNN on CIFAR-10 data (BMUF)')
plt.xlabel("Number of epoch")
plt.ylabel("Training error in %")
plt.savefig(ERROR_GRAPH_OUTPUT_PATH + '/multiples/7/error-128,1000,1280,2000,12800.pdf')

# PLOT TIME GRAPH
fig, ax = plt.subplots()
ax.plot(list(range(1, len(time_lists[0][1:]) + 1)), time_lists[0][1:], color='red', label='Sync period:128')
ax.plot(list(range(1, len(time_lists[1][1:]) + 1)), time_lists[1][1:], color='black', label='Sync period:1000')
ax.plot(list(range(1, len(time_lists[2][1:]) + 1)), time_lists[2][1:], color='green', label='Sync period:1280')
ax.plot(list(range(1, len(time_lists[3][1:]) + 1)), time_lists[3][1:], color='magenta', label='Sync period:2000')
ax.plot(list(range(1, len(time_lists[4][1:]) + 1)), time_lists[4][1:], color='blue', label='Sync period:12800')
ax.legend(loc='center right')
# ADD AVERAGE TIME
avg_text = ''
avg_text += 'Avg 128: ' + str(np.average(time_lists[0])) + '\n'
avg_text += 'Avg 1000: ' + str(np.average(time_lists[1])) + '\n'
avg_text += 'Avg 1280: ' + str(np.average(time_lists[2])) + '\n'
avg_text += 'Avg 2000: ' + str(np.average(time_lists[3])) + '\n'
avg_text += 'Avg 12800: ' + str(np.average(time_lists[4])) + '\n'
plt.text(0.2, 0.5, avg_text,
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.title('Time per epoch of Convolutional NN on CIFAR-10 data')
plt.xlabel("Number of epoch")
plt.ylabel("Time in second")
plt.savefig(TIME_GRAPH_OUTPUT_PATH + '/multiples/7/time-128,1000,1280,2000,12800.pdf')

# PLOT TIME x Error Graph
fig, ax = plt.subplots()
accum_time = []
for i in range(5):
    accum_time.append([0])

for j in range(5):
    for i in time_lists[j]:
        accum_time[j].append(float(i) + accum_time[j][-1])

for i in range(5):
    accum_time[i] = accum_time[i][1:]

ax.plot(accum_time[0], error_lists[0], color='red', label='Sync period:128')
ax.plot(accum_time[1], error_lists[1], color='black', label='Sync period:1000')
ax.plot(accum_time[2], error_lists[2], color='green', label='Sync period:1280')
ax.plot(accum_time[3], error_lists[3], color='magenta', label='Sync period:2000')
ax.plot(accum_time[4], error_lists[4], color='blue', label='Sync period:12800')
ax.legend(loc='upper right')
plt.title('Training Error by Time of CNN on CIFAR-10 data (BMUF)')
plt.xlabel("Time")
plt.ylabel("Training error in %")
plt.savefig(ERROR_GRAPH_OUTPUT_PATH + '/multiples/7/error-by-time-128,1000,1280,2000,12800.pdf')
'''
