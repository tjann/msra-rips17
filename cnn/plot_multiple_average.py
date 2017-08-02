import collections
import sys
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ERROR_GRAPH_OUTPUT_PATH = 'error_graphs'
TIME_GRAPH_OUTPUT_PATH = 'time_graphs'

file_names = ['128.txt', '1280.txt', '12800.txt']

def parse_output_file(file_name):
    error_big_list = []
    time_big_list = []
    for k in range(1,6):
        with open('log/6/res-'+ str(k) + '-' + file_name, 'r') as f:
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
        error_big_list.append(error_list)
        time_big_list.append(time_list)
    return (np.average(error_big_list, axis=0), np.average(time_big_list, axis=0))

error_lists = [parse_output_file(i)[0] for i in file_names]
time_lists = [parse_output_file(i)[1] for i in file_names]

# Check if there is any funky value in the list
for i in range(len(error_lists)):
    for j in range(len(error_lists[i])):
        if error_lists[i][j] > 100:
            print('ERROR i: ', i, 'j: ', j, 'value: ', error_lists[i][j])
        if time_lists[i][j] > 60:
            print('TIME i: ', i, 'j: ', j, 'value: ', time_lists[i][j])

print('ERROR: ')
print(error_lists)
print('TIME: ')
print(time_lists)

# ERROR i:  0 j:  40 value:  1000013.3112
# ERROR i:  1 j:  27 value:  1000018.1116
# ERROR i:  1 j:  35 value:  1000015.4432
# ERROR i:  2 j:  96 value:  100001.3888
error_lists[0][40] = 13.3112
error_lists[1][27] = 18.1116
error_lists[1][35] = 15.4432
error_lists[2][96] = 1.3888
# PLOT ERROR GRAPH
fig, ax = plt.subplots()
ax.plot(list(range(1, len(error_lists[0]) + 1)), error_lists[0], color='red', label='Sync period:128')
ax.plot(list(range(1, len(error_lists[1]) + 1)), error_lists[1], color='green', label='Sync period:1280')
ax.plot(list(range(1, len(error_lists[2]) + 1)), error_lists[2], color='blue', label='Sync period:12800')
#ax.plot(list(range(1, len(error_lists[3]) + 1)), error_lists[3], color='black', label='Sync period:256')
#ax.plot(list(range(1, len(error_lists[4]) + 1)), error_lists[4], color='orangered', label='Sync period:512')
#ax.plot(list(range(1, len(error_lists[5]) + 1)), error_lists[5], color='darkmagenta', label='Sync period:1024')
ax.legend(loc='upper right')
plt.title('Training Error of CNN (averaged 5 times) on CIFAR-10 data')
plt.xlabel("Number of epoch")
plt.ylabel("Training error in %")
plt.savefig(ERROR_GRAPH_OUTPUT_PATH + '/multiples/6/error-128,1280,12800.png')

# PLOT TIME GRAPH
fig, ax = plt.subplots()
ax.plot(list(range(1, len(time_lists[0][1:]) + 1)), time_lists[0][1:], color='red', label='Sync period:128')
ax.plot(list(range(1, len(time_lists[1][1:]) + 1)), time_lists[1][1:], color='green', label='Sync period:1280')
ax.plot(list(range(1, len(time_lists[2][1:]) + 1)), time_lists[2][1:], color='blue', label='Sync period:12800')
#ax.plot(list(range(1, len(error_lists[3]) + 1)), error_lists[3], color='black', label='Sync period:256')
#ax.plot(list(range(1, len(error_lists[4]) + 1)), error_lists[4], color='orangered', label='Sync period:512')
#ax.plot(list(range(1, len(error_lists[5]) + 1)), error_lists[5], color='darkmagenta', label='Sync period:1024')
ax.legend(loc='upper right')
# ADD AVERAGE TIME
avg_text = ''
avg_text += 'Avg 128: ' + str(np.average(time_lists[0])) + '\n'
avg_text += 'Avg 1280: ' + str(np.average(time_lists[1])) + '\n'
avg_text += 'Avg 12800: ' + str(np.average(time_lists[2])) + '\n'
plt.text(0.2, 0.9, avg_text,
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.title('Time per epoch of CNN (averaged 5 times) on CIFAR-10 data')
plt.xlabel("Number of epoch")
plt.ylabel("Time in second")
plt.savefig(TIME_GRAPH_OUTPUT_PATH + '/multiples/6/time-128,1280,12800.png')
