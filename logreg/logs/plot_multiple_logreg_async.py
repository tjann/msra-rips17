import collections
import sys
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


#############################################################
###################### VARIABLE INPUTS ######################
synchro = 'async'
num_sf = 3
file_names100 = ['async100a.txt', 'async100b.txt', 'async100c.txt', 'async100d.txt', 'async100e.txt']
file_names1000 = ['async1000a.txt', 'async1000b.txt', 'async1000c.txt', 'async1000d.txt', 'async1000e.txt']
file_names10000 = ['async10000a.txt', 'async10000b.txt', 'async10000c.txt', 'async10000d.txt', 'async10000e.txt']

num_epoch = 149  # num_samples for logreg
num_workers = 8
#############################################################
#############################################################


GRAPH_OUTPUT_PATH = '../' + synchro + '_graphs'

def parse_output_file(file_name):
    with open('../logs/' + file_name, 'r') as f:
        list_of_epoch_information = [l for l in f]
        list_of_epoch_information = [l for l in list_of_epoch_information if re.match(r'.+Epoch \d+\. Worker \d+ set ', l)]
   # print(len(list_of_epoch_information))

    rowsOfErrors = []
    rowsOfTimes = []
    for i in range(num_workers):
        rowsOfErrors.append([]);
        rowsOfTimes.append([]);

    for line in (list_of_epoch_information):
        worker = int(re.search(r'.+Worker (\d+)', line).group(1))
        error = float(re.search(r'.+train loss (0\.\d+)', line).group(1))
        print(file_name)
        time = float(re.search(r'.+average computation time (\d+\.\d+)', line).group(1))
        rowsOfErrors[worker].append(error)
        rowsOfTimes[worker].append(time)
    
    avgWorkerErrors = np.average(rowsOfErrors, axis=0)
    avgWorkerTimes = np.average(rowsOfTimes, axis=0)
    return avgWorkerErrors, avgWorkerTimes

def err_and_time_for_all_sf():
    error_lists = []
    time_lists = []
    rowsOfTrialErrors = [parse_output_file(i)[0] for i in file_names100]
    rowsOfTrialTimes = [parse_output_file(i)[1] for i in file_names100]
    error_lists.append(np.average(rowsOfTrialErrors, axis=0))
    time_lists.append(np.average(rowsOfTrialTimes, axis=0))
    rowsOfTrialErrors = [parse_output_file(i)[0] for i in file_names1000]
    rowsOfTrialTimes = [parse_output_file(i)[1] for i in file_names1000]
    error_lists.append(np.average(rowsOfTrialErrors, axis=0))
    time_lists.append(np.average(rowsOfTrialTimes, axis=0))
    rowsOfTrialErrors = [parse_output_file(i)[0] for i in file_names10000]
    rowsOfTrialTimes = [parse_output_file(i)[1] for i in file_names10000]
    error_lists.append(np.average(rowsOfTrialErrors, axis=0))
    time_lists.append(np.average(rowsOfTrialTimes, axis=0))
    return error_lists, time_lists
#print(errors)
#print(times)

error_lists, time_lists = err_and_time_for_all_sf()
color_list = ['red', 'green', 'blue', 'magenta', 'black']
label_list = ['Communication period: 100', 'Communication period: 1000', 'Communication period: 10000']
avg_list = ['Avg 100', 'Avg 1000', 'Avg 10000']


# No sanity checks implemented for errors or times
fig, ax = plt.subplots()
# PLOT SAMPLES SEEN X TRAIN LOSS GRAPH
for i in range(num_sf):
    x = list(range(100032, 14963129, 100032))
#    x = x[:30]
    y = list(error_lists[i])
#    y = y[:30]
    # ax.plot(list(range(100032, 14963129, 100032)), list(error_lists[i]), color=color_list[i], label=label_list[i])
    ax.plot(x, y, color=color_list[i], label=label_list[i])
ax.legend(loc='upper right', fontsize=14)
#plt.title('Train Loss of Logistic Regression on KDD Cup 2012 Data (' + synchro + ')')
plt.xlabel("Number of Samples Seen", fontsize=18)
plt.ylabel("Train Loss", fontsize=18)
plt.savefig(GRAPH_OUTPUT_PATH + '/error_graph_sf100,1000,10000.pdf')


# PLOT SAMPLES SEEN X TIME GRAPH
fig, ax = plt.subplots()
for i in range(num_sf):
    ax.plot(list(range(100032, 14963129, 100032)), list(time_lists[i]), color=color_list[i], label=label_list[i])
ax.legend(loc='upper right', fontsize=14)
# add average time
avg_text = ''
for i in range(num_sf):
    avg_text += avg_list[i] + ': ' +str(np.average(time_lists[i])) + '\n'
plt.text(0.2, 0.9, avg_text,
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
#plt.title('Time Per Samples Set Logistic Regression on KDD Cup 2012 Data (' + synchro + ')')
plt.xlabel("Number of Samples Seen", fontsize=18)
plt.ylabel("Compute Time (s)", fontsize=18)
plt.savefig(GRAPH_OUTPUT_PATH + '/time_graph_sf100,1000,10000.pdf')

# PLOT TIME x TRAIN LOSS
fig, ax = plt.subplots()
print('ax is: ', ax)
accum_time = []
for i in range(num_sf):
    accum_time.append([0])
for j in range(num_sf):
    for i in time_lists[j]:
        accum_time[j].append(float(i) + accum_time[j][-1])
for i in range(num_sf):
    accum_time[i] = accum_time[i][1:]
for i in range(num_sf):
    ax.plot(list(accum_time[i]), list(error_lists[i]), color=color_list[i], label=label_list[i])
ax.legend(loc='upper right', fontsize=14)
#plt.title('Train Loss by Time of Logistic Regression on KDD Cup 2012 Data (' + synchro + ')')
plt.xlabel("Time (s)", fontsize=18)
plt.ylabel("Train Loss", fontsize=18)
plt.savefig(GRAPH_OUTPUT_PATH + '/error_by_time_graph_sf100,1000,10000.pdf')
"""
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

"""
