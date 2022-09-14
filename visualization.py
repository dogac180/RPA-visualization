#!/usr/bin/env python
# coding: utf-8

# In[1]:


from inspect import Parameter
from turtle import color
from unittest import result
from isort import code, file
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
from datetime import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from itertools import *
import collections

# Read Excel Files
all_jobs = pd.read_excel("All_Jobs1.xlsx", sheet_name="Sayfa1")
all_queue_items = pd.read_excel("LastQueue.xlsx", sheet_name="Sayfa1")

# create a list containing the name of unique tasks
items = all_queue_items.ExecutorJobId.unique().tolist()
items.remove(11018.0)
items_list = sorted(items)

# create a dataframe using the file QueueItems_Full and make necessary adjustments
all_queue_df = all_queue_items[
    all_queue_items["ExecutorJobId"].isin(items_list)
].sort_values(by="ExecutorJobId", ascending=True)
all_queue_df = all_queue_df.dropna(subset=["ExecutorJobId"], how="all")

# create a dataframe using the file All_Jobs1 and make necessary adjustments
all_jobs_df = all_jobs[all_jobs["Id"].isin(items_list)].sort_values(
    by="Id", ascending=True
)
all_jobs_df = all_jobs_df[:-1]

# create periods
starting = time(9, 0, 0)
mid = time(12, 30, 0)
afternoon = time(15, 30, 0)
ending = time(18, 30, 0)

# converting time columns to datetime
all_queue_df["StartProcessing"] = pd.to_datetime(
    all_queue_df["StartProcessing"], format="%d/%m/%Y"
)
all_queue_df["EndProcessing"] = pd.to_datetime(
    all_queue_df["EndProcessing"], format="%d/%m/%Y"
)
all_jobs_df["CreationTime"] = pd.to_datetime(
    all_jobs_df["CreationTime"], format="%d/%m/%Y"
)

all_jobs_df["creation_time"] = [d.time() for d in all_jobs_df["CreationTime"]]
all_queue_df["new_date_start"] = [d.date() for d in all_queue_df["StartProcessing"]]
all_queue_df["new_time_start"] = [d.time() for d in all_queue_df["StartProcessing"]]
all_queue_df["new_date_end"] = [d.date() for d in all_queue_df["EndProcessing"]]
all_queue_df["new_time_end"] = [d.time() for d in all_queue_df["EndProcessing"]]

scheduled_jobs = {}
manuel_jobs = {}
total_number_of_jobs_under_processes_periods = {}
def create_empty_dics():
    for i in range(len(all_queue_df)):
        source_item = all_queue_df.iloc[i]["Source"]
        process_item = all_queue_df.iloc[i]["ProcessID"]
        Job_item = all_queue_df.iloc[i]["ExecutorJobId"]
        total_number_of_jobs_under_processes_periods[process_item,Job_item] = [0,0,0]
        if source_item == 1:
            scheduled_jobs[
                process_item,
                Job_item
            ] = [0, 0, 0]
        elif source_item == 0:
            manuel_jobs[
                process_item,
                Job_item
            ] = [0, 0, 0]
    return scheduled_jobs, manuel_jobs, total_number_of_jobs_under_processes_periods

scheduled_jobs, manuel_jobs, total_number_of_jobs_under_processes_periods = create_empty_dics()


# In[2]:


def create_schedule_manuel_dics():
    for i in range(len(all_queue_df)):
        process_id = all_queue_df.iloc[i]["ProcessID"]
        job_id = all_queue_df.iloc[i]["ExecutorJobId"]
        source = all_queue_df.iloc[i]["Source"]
        start_time = all_queue_df.iloc[i]["new_time_start"]
        end_time = all_queue_df.iloc[i]["new_time_end"]

        if source == 1:
            if end_time < starting or end_time > ending:
                scheduled_jobs[process_id,job_id][2] += 1
            
            if end_time >= starting and end_time <= mid:
                scheduled_jobs[process_id,job_id][0] += 1
            
            if end_time > mid and end_time <= ending:
                scheduled_jobs[process_id,job_id][1] += 1
        
        else:
            if end_time < starting or end_time > ending:
                manuel_jobs[process_id,job_id][2] += 1
            
            if end_time >= starting and end_time <= mid:
                manuel_jobs[process_id,job_id][0] += 1
            
            if end_time > mid and end_time <= ending:
                manuel_jobs[process_id,job_id][1] += 1
    
    return scheduled_jobs, manuel_jobs

scheduled_jobs_under_process, manuel_jobs_under_process = create_schedule_manuel_dics()


# In[3]:


print(scheduled_jobs_under_process)


# In[4]:


print(manuel_jobs_under_process)


# In[5]:


width = 0.35
x = ["(9:00-12:30)", "(12:30-15:30)", "(18:30-00:00)"]
for k, v in manuel_jobs_under_process.items():
    bar1 = np.arange(len(x))
    plt.bar(bar1, v, width, label="Number of Jobs", color="green")
    plt.xticks(bar1, x)
    plt.title("Manually Executed Jobs "+str(k[1])+" in "+str(k[0]), fontsize=13, fontweight="bold")
    plt.ylabel("Number of Jobs",fontsize=10)
    plt.legend()
    plt.show()


# In[6]:



def draw_err(scheduled_jobs_under_process):
    width = 0.35
    x = ["(9:00-12:30)", "(12:30-18:30)", "(18:30-00:00)"]
    for k, v in scheduled_jobs_under_process.items():
        bar1 = np.arange(len(x))
        plt.bar(bar1, v, width, label="Number of Jobs", color="green")
        plt.xticks(bar1, x)
        plt.title("Scheduled Jobs "+str(k[1])+" in "+str(k[0]), fontsize=13, fontweight="bold")
        plt.ylabel("Number of Jobs",fontsize=10)
        plt.legend()
        plt.show()
    


# In[7]:


# how many scheduled and manual for each process
total_scheduled_and_manual = {}
for k,v in scheduled_jobs_under_process.items():
    if k[0] in total_scheduled_and_manual.keys():
        total_scheduled_and_manual[k[0]][0] += sum(v)
    else:  
        total_scheduled_and_manual[k[0]] = [sum(v),0]

for k,v in manuel_jobs_under_process.items():
    if k[0] in total_scheduled_and_manual.keys():
        total_scheduled_and_manual[k[0]][1] +=  sum(v)
    else:
        total_scheduled_and_manual[k[0]] = [0,sum(v)]
print(total_scheduled_and_manual)


# In[8]:


def draw_scheduled_manual_for_processes():
    width = 0.35
    x = ["Scheduled", "Manual"]
    for k, v in total_scheduled_manual_for_processes.items():
        bar1 = np.arange(len(x))
        plt.bar(bar1, v, width, label="Number of Jobs", color="orange")
        plt.xticks(bar1, x)
        plt.title("Manual and Scheduled Jobs under "+str(k), fontsize=13, fontweight="bold")
        plt.xlabel("Execution Types",fontsize=10)
        plt.ylabel("Number of Jobs",fontsize=10)
        plt.legend()
        plt.show()


# In[9]:


dic_system = {}
dic_statue = {}
system_ = {}
statue_ = {}
success_ = {}
for i in range(len(all_queue_df)):
    id_from_ = all_queue_df.iloc[i]["ExecutorJobId"]
    process_id_ = all_queue_df.iloc[i]["ProcessID"]
    dic_system[process_id_,id_from_] = [0, 0, 0]
    dic_statue[process_id_,id_from_] = [0, 0, 0]
    system_[process_id_,id_from_] = [0,0,0]
    statue_[process_id_,id_from_] = [0,0,0]
    success_[process_id_,id_from_] = [0,0,0]


# In[10]:


# create a dictionary with process and jobs as keys
for i in range(len(all_queue_df)):
    process_id = all_queue_df.iloc[i]["ProcessID"]
    job_id = all_queue_df.iloc[i]["ExecutorJobId"]
    system_statue = all_queue_df.iloc[i]["ProcessingExceptionType"]
    end_time = all_queue_df.iloc[i]["new_time_end"]
    if end_time < starting or end_time > ending:
        total_number_of_jobs_under_processes_periods[process_id,job_id][2] += 1
    elif end_time >= starting and end_time <= mid:
        total_number_of_jobs_under_processes_periods[process_id,job_id][0] += 1
    elif end_time > mid and end_time <= ending:
        total_number_of_jobs_under_processes_periods[process_id,job_id][1] += 1
        
    if system_statue == 1:
        if end_time < starting or end_time > ending:
            dic_statue[process_id,job_id][2] += 1

        if end_time >= starting and end_time <= mid:
            dic_statue[process_id,job_id][0] += 1

        if end_time > mid and end_time <= ending:
            dic_statue[process_id,job_id][1] += 1
            
    if system_statue == 0:
        if end_time < starting or end_time > ending:
            dic_system[process_id,job_id][2] += 1

        if end_time >= starting and end_time <= mid:
            dic_system[process_id,job_id][0] += 1

        if end_time > mid and end_time <= ending:
            dic_system[process_id,job_id][1] += 1

print(dic_system)


# In[11]:


print(dic_statue)


# In[12]:


print(total_number_of_jobs_under_processes_periods)


# In[13]:


# total status and system errors for each period
total_system_error = {}
total_statue_error = {}
for k,v in dic_statue.items():
    if k[0] in total_statue_error.keys():
        total_statue_error[k[0]][0] += v[0]
        total_statue_error[k[0]][1] += v[1]
        total_statue_error[k[0]][2] += v[2]
    else:  
        total_statue_error[k[0]] = [v[0],v[1],v[2]]

for k,v in dic_system.items():
    if k[0] in total_system_error.keys():
        total_system_error[k[0]][0] += v[0]
        total_system_error[k[0]][1] += v[1]
        total_system_error[k[0]][2] += v[2]
    else:  
        total_system_error[k[0]] = [v[0],v[1],v[2]]


# In[14]:


# draw a graph for the number of system and status errors for periods
print(total_system_error)


# In[15]:


a = list(total_system_error.values())
b = list(total_statue_error.values())
c = list(total_statue_error.keys())
barWidth = 0.25
#fig = plt.subplots(figsize =(12, 8))
#set position of bar x axis
br1 = np.arange(3)
br2 = [x + barWidth for x in br1]

#make the plot
for i in range(len(a)):
    plt.bar(br1, a[i], width = barWidth,
        edgecolor ='grey', label ='System Error')

    plt.bar(br2, b[i], color ='g', width = barWidth,
        edgecolor ='grey', label ='Status Error')

# Adding Xticks
    plt.xlabel('', fontweight ='bold', fontsize = 15)
    plt.ylabel('Number of Failed Tasks', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(3)],
        ['(9:00-12:30)','(12:30-18:30)','(18:30-00:00)'])
    plt.title('System & Status Errors For Each Process '+str(c[i]), fontweight ='bold', fontsize = 15)
    plt.legend()
    plt.show()


# In[16]:


# calculate the accurcy for each period
for i in range(len(all_queue_df)):
    process_id = all_queue_df.iloc[i]["ProcessID"]
    job_id = all_queue_df.iloc[i]["ExecutorJobId"]
    system_statue = all_queue_df.iloc[i]["ProcessingExceptionType"]
    end_time = all_queue_df.iloc[i]["new_time_end"]
    

    if end_time >= starting and end_time <= mid:
        success_[process_id,job_id][0] = (
            (
                total_number_of_jobs_under_processes_periods[process_id,job_id][0]
                - dic_system[process_id,job_id][0]
                - dic_statue[process_id,job_id][0]
            )
            / total_number_of_jobs_under_processes_periods[process_id,job_id][0]
        ) * 100
        system_[process_id,job_id][0] = (
            dic_system[process_id,job_id][0]
            / total_number_of_jobs_under_processes_periods[process_id,job_id][0]
        ) * 100
        statue_[process_id,job_id][0] = (
            dic_statue[process_id,job_id][0]
            / total_number_of_jobs_under_processes_periods[process_id,job_id][0]
        ) * 100

    elif end_time > mid and end_time <= ending:

        success_[process_id,job_id][1] = (
            (
                total_number_of_jobs_under_processes_periods[process_id,job_id][1]
                - (dic_system[process_id,job_id][1] + dic_statue[process_id,job_id][1])
            )
            / total_number_of_jobs_under_processes_periods[process_id,job_id][1]
        ) * 100
        system_[process_id,job_id][1] = (
            dic_system[process_id,job_id][1]
            / total_number_of_jobs_under_processes_periods[process_id,job_id][1]
        ) * 100
        statue_[process_id,job_id][1] = (
            dic_statue[process_id,job_id][1]
            / total_number_of_jobs_under_processes_periods[process_id,job_id][1]
        ) * 100

    elif (
        end_time < starting or end_time > ending
    ):  # or can be changed to and, then delete if condition
        #print(number_of_jobs_for_each_process_in_periods[process_id][2])
        if total_number_of_jobs_under_processes_periods[process_id,job_id][2] != 0:

            success_[process_id,job_id][2] = (
                (
                    total_number_of_jobs_under_processes_periods[process_id,job_id][2]
                    - dic_system[process_id,job_id][2]
                    - dic_statue[process_id,job_id][2]
                )
                / total_number_of_jobs_under_processes_periods[process_id,job_id][2]
            ) * 100
            system_[process_id,job_id][2] = (
                dic_system[process_id,job_id][2]
                / total_number_of_jobs_under_processes_periods[process_id,job_id][2]
            ) * 100
            statue_[process_id,job_id][2] = (
                dic_statue[process_id,job_id][2]
                / total_number_of_jobs_under_processes_periods[process_id,job_id][2]
            ) * 100


# In[17]:


print(total_number_of_jobs_under_processes_periods)


# In[18]:


total_workload_for_processes = {}    
for k,v in total_number_of_jobs_under_processes_periods.items():
    if k[0] in total_workload_for_processes.keys():
        total_workload_for_processes[k[0]] += sum(v)
    else:  
        total_workload_for_processes[k[0]] = sum(v)
print(total_workload_for_processes)


# In[19]:


print(success_)


# In[20]:


print(statue_)


# In[21]:


print(system_)


# In[22]:


# for process in range(len(process_id_list)):
width = 0.35
label_periods = ["(9-12:30)", "(12:30-18:30)", "(18:30-00:00)"]
def draw():
    for i in success_.keys():
        job_success = success_[i]
        job_system_error = system_[i]
        job_statue_error = statue_[i]
        process_name = i
        # [job_success[0],job_system_error[0],job_statue_error[0]],
        plotdata = pd.DataFrame(
            {
                "Success": job_success,
                "System_Error_Rate": job_system_error,
                "Statue_Error_Rate": job_statue_error,
            },
            index=label_periods,
        )
        print(plotdata)
        stacked_data = plotdata #.apply(lambda x: x * 100 / sum(x), axis=1)
        stacked_data.plot(kind="bar", stacked=True)
        
        plt.title("The Performance Metrics for "+str(process_name),fontsize=14,fontweight="bold")
        
        #plt.title("Process", str(process_name), "Performance(%)", fontsize=14, fontweight="bold")
        plt.xlabel("Periods", fontsize=11)
        plt.ylabel("Rate", fontsize=11)
        plt.show()
print(draw())


# In[23]:


# take the average of the success, sys, stat for each process
succ_average = {}
sys_average = {}
stat_average = {}
for k,v in success_.items():
    if k[0] in succ_average.keys():
        if v.count(0) >= 2:
            succ_average[k[0]] += sum(v)
        if v.count(0) == 1:
            succ_average[k[0]] += (sum(v)/(2*len(success_)))
        if v.count(0) == 0:
            succ_average[k[0]] += (sum(v)/(3*len(success_)))

print(succ_average)


# In[25]:


# create robot dic
robot_names = all_jobs_df.RobotName.unique().tolist()
print(robot_names)


# In[35]:


all_jobs_df["StartTime"] = pd.to_datetime(
    all_jobs_df["StartTime"], format="%d/%m/%Y"
)
all_jobs_df["EndTime"] = pd.to_datetime(
    all_jobs_df["EndTime"], format="%d/%m/%Y"
)
all_jobs_df["new_time_end"] = [d.time() for d in all_jobs_df["EndTime"]]
all_jobs_df["new_time_start"] = [d.time() for d in all_jobs_df["StartTime"]]

robots_duration = {}
for i in robot_names:
    robots_duration[i] = 0

for i in range(len(all_jobs_df)):
    robot = all_jobs_df.iloc[i]["RobotName"]
    start = all_jobs_df.iloc[i]["StartTime"]
    end = all_jobs_df.iloc[i]["EndTime"]
    #print(end-start)
    
    robots_duration[robot] += (end - start).total_seconds() / 60.0
    
print(robots_duration)


# In[37]:


accuracy_list = list(robots_duration.values())
acc_key_list = list(robots_duration.keys())
plt.style.use('ggplot')
plt.barh(acc_key_list, accuracy_list, align='center')
plt.title("The Total Working Minutes For Each Robot", fontweight ='bold', fontsize=15)
#plt.xlabel('The Total Working Time in Minutes', fontsize=11)
#plt.ylabel('The Accuracy',fontweight ='bold', fontsize=15)
plt.show()


# In[1]:





# In[ ]:




