import collections
import sys
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ERROR_GRAPH_OUTPUT_PATH = 'error_graphs/sync'
TIME_GRAPH_OUTPUT_PATH = 'time_graphs/sync'

if len(sys.argv) != 3:
    print("Please supply exactly one file to be parsed")
    sys.exit(1)

file_name = sys.argv[1]
target_name = sys.argv[2]

with open('logs/sync/' + file_name, 'r') as f:
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
epoch_list = list(range(len(error_epoch)))
for j in error_epoch:
    error_list.append(error_epoch[j])
    time_list.append(time_epoch[j])
    # FOR DEBUG: print(j, ': ', error_epoch[j])

# PLOT GRAPH FOR ERROR
avg_error = np.average(error_list)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(epoch_list, error_list)
plt.title('Training Error for Single Worker')
plt.xlabel('Number of epoch')
plt.ylabel('Error rate in %')
plt.text(0.5, 0.9,'Average error: ' + str(avg_error) + '%', ha='center', va='center', transform=ax.transAxes, bbox=dict(facecolor='blue', alpha=0.5))
plt.savefig(ERROR_GRAPH_OUTPUT_PATH + '/' + 'error-' + target_name + '.png')

# PLOT GRAPH FOR TIME
total_time = str(sum(time_list))
average_time = str(np.average(time_list))
fig = plt.figure()
ax = fig.add_subplot(111)
time_text = 'Total time: ' + total_time + ' s\n'
time_text += 'Average time: ' + average_time + ' s\n'
plt.plot(epoch_list, time_list)
plt.title('Time Required for Single Worker')
plt.xlabel('Number of epoch')
plt.ylabel('Time in second')
plt.text(0.5, 0.9, time_text, ha='center', va='center', transform=ax.transAxes, bbox=dict(facecolor='blue', alpha=0.5))
plt.savefig(TIME_GRAPH_OUTPUT_PATH + '/' + 'time-' + target_name + '.png')
