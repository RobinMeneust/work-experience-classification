import json
from datetime import datetime
import os
from sentence_transformers.losses import CosineSimilarityLoss, BatchAllTripletLoss
import time
import numpy as np


#############################################
# Save files
#############################################

def save_to_json(results, train_times, eval_times, params, folder_path):
    """Save test data in the given folder
        The filename of the save files are the current time, e.g. 2024-03-14_13-58-01
    Args:
        results (dict): Test results (F1-scores) for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of F1-scores)
        run_times (dict): Run times for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of the training run times)
        params (dict): Parameters used for the test (N-shots, number of iterations, ...)
        folder_path (string): Path were the test file will be saved
    """
    
    if "loss" in params:
        if type(params["loss"]) == type({}):
            list_losses = []
            for key in params["loss"].keys():
                list_losses.append(key)
            params["loss"] = list_losses
        elif type(params["loss"]) != type(""):
            if params["loss"] == CosineSimilarityLoss:
                params["loss"] = "Pair-wise"
            elif params["loss"] == BatchAllTripletLoss:
                params["loss"] = "Triplet"
            else:
                params["loss"] = "UNKNOWN"

    if "distance" in params:
        list_distances = []
        for key in params["distance"].keys():
            list_distances.append(key)
            params["distance"] = list_distances

    if "num_epochs" in params:
        if type(params["num_epochs"]) == type([]):
            for i in range(len(params["num_epochs"])):
                params["num_epochs"][i] = f"({params['num_epochs'][i][0]},{params['num_epochs'][i][1]})"
        else:
            params["num_epochs"] = f"({params['num_epochs'][0]},{params['num_epochs'][1]})"

    object = {"results":results, "train_times":train_times, "eval_times":eval_times, "params": params}

    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Generate file name
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = folder_path+'/'+str(date)+".json"

    # Create file and save data
    with open(file_name, 'w') as file:
        json.dump(object, file)
   
#############################################
# Get remaining time
#############################################   
     
def get_remaining_time_str(start_time, progress, progress_end):
    """Get the remaining time as a string (in seconds if < 60 s otherwise in minutes, e.g: 67 m)

    Args:
        start_time (float): Time when the first iteration has started
        progress (int): Current iteration (start at 1)
        progress_end (int): Number of iterations

    Returns:
        string: String representing the estimated remaining time in minutes (if at least 60 s) or in seconds
    """
    elapsed_time = time.time() - start_time
    if progress <= 1:
        return "?"
    
    time_per_iter = elapsed_time / progress
    remaining_seconds = (time_per_iter * (1 + progress_end - progress))
    if remaining_seconds < 60:
        return str(round(remaining_seconds)) + " s"
    else:
        return str(round(remaining_seconds/60)) + " m"
    

#############################################
# Load tests data
#############################################   

def load_results_data(filename, folder):
    """Load test results from one file in the given folder, whose filename is provided

    Args:
        filename (string): Name of the file
        folder (string): Name of the folder where the file is

    Returns:
        dict: Test results (F1-scores) for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of F1-scores)
        dict: Run times for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of the training run times)
        dict: Params of the test
    """
    with open(folder+"/"+filename, 'r') as file:
        data = json.load(file)
    
    train_times = data["train_times"] if "train_times" in data else {}
    eval_times = data["eval_times"] if "eval_times" in data else {}
    
    return data['results'], train_times, eval_times, data['params']

def load_latest_results_data(folder):
    """Load the latest test results from one folder

    Args:
        folder (string): Name of the folder where the file is

    Returns:
        dict: Test results (F1-scores) for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of F1-scores)
        dict: Run times for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of the training run times)
        dict: Params of the test
    """
    filenames = os.listdir(folder)
    latest = max(filenames, key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    return load_results_data(latest, folder)

def load_all_results_data(folder, test_name, filters={}):
    """Load, aggregate and sort all tests results in the given folder with search filters.

    Args:
        folder (string): Name of the folder where the file is
        test_name (string): Name of the test (needed to agregate data)
        filters (dict, optional): Search filters (e.g {"n_max_iter_per_shot": 10}). Defaults to {}.

    Returns:
        dict: Test results (F1-scores) for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of F1-scores)
        dict: Run times for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of the training run times)
        dict: Params of the test
    """
    
    filenames_list = os.listdir(folder)
    all_data = {"results":{}, "train_times":{}, "eval_times":{}}
 
 
    for filename in filenames_list:
        do_add_data = True
        tested_param = test_name
        new_data = {"results":{}, "train_times":{}, "eval_times":{}}
  
        with open(folder+"/"+filename, 'r') as file:
            data = json.load(file)
                            
            for param_key, param_value in data["params"].items():
                if not (param_key in filters):
                    # No filter
                    if param_key == tested_param:
                        new_data["results"] = data["results"]
                        if "train_times" in data:
                            new_data["train_times"] = data["train_times"]
                        if "eval_times" in data:
                            new_data["eval_times"] = data["eval_times"]
                    continue
                
                if type(filters[param_key]) != type([]) and type(filters[param_key]) != type({}):
                    filters[param_key] = [filters[param_key]]

                if type(param_value) == type([]) and len(param_value) > 0 and type(param_value[0]) == type([]):
                    for i in range(len(param_value)):
                        param_value[i] = f"[{param_value[i][0]},{param_value[i][1]}]"

                if type(param_value) == type({}) and type(filters[param_key]) == type({}):
                    for sub_filter_key, sub_filter_val in filters[param_key].items():
                        if not (sub_filter_key in param_value) or param_value[sub_filter_key] != sub_filter_val:
                            do_add_data = False
                            break
                    continue
                

                if param_key == tested_param:
                    for filter_value in filters[param_key]:
                        if type(filter_value) == type(()) and (len(param_value) == 2 and type(param_value[0]) == type("")):
                            filter_value = f"({filter_value[0]},{filter_value[1]})"
                            
                        if type(filter_value) == type([]) and len(filter_value) == 2:
                            filter_value = f"[{filter_value[0]},{filter_value[1]}]"
                            
                        if filter_value in param_value and str(filter_value) in data["results"]:
                            new_data["results"][filter_value] = data["results"][str(filter_value)]
                                                        
                            if "train_times" in data:
                                new_data["train_times"][filter_value] = data["train_times"][str(filter_value)]
                            if "eval_times" in data:
                                new_data["eval_times"][filter_value] = data["eval_times"][str(filter_value)]
                elif not (param_value in filters[param_key]):
                    do_add_data = False
                    break
 
        if do_add_data == True:
            for output_type in new_data.keys(): # results and run_times
                for key in new_data[output_type].keys():
                    if key in all_data[output_type]:
                        all_data[output_type][key] = np.concatenate((all_data[output_type][key], new_data[output_type][key]))
                    else:
                        all_data[output_type][key] = new_data[output_type][key]
    isSorted = False   
    try:
        # Try to sort the keys if they are number
        all_data["results"] = dict(sorted(all_data["results"].items(), key=lambda x: float(x[0])))
        all_data["train_times"] = dict(sorted(all_data["train_times"].items(), key=lambda x: float(x[0])))
        all_data["eval_times"] = dict(sorted(all_data["eval_times"].items(), key=lambda x: float(x[0])))
        isSorted = True
    except:""
    if not isSorted:
        try:
            # Try to sort the keys if they are pairs of numbers (tuples or lists)
            all_data["results"] = dict(sorted(all_data["results"].items(), key=lambda x: (float(json.loads(x[0])[0]),float(json.loads(x[0])[1]))))
            all_data["train_times"] = dict(sorted(all_data["train_times"].items(), key=lambda x: (float(json.loads(x[0])[0]),float(json.loads(x[0])[1]))))
            all_data["eval_times"] = dict(sorted(all_data["eval_times"].items(), key=lambda x: (float(json.loads(x[0])[0]),float(json.loads(x[0])[1]))))
            isSorted = True
        except:""
    if not isSorted:
        try:
            # Try to sort the keys if they are of another type (e.g strings)
            all_data["results"] = dict(sorted(all_data["results"].items(), key=lambda x: x[0]))
            all_data["train_times"] = dict(sorted(all_data["train_times"].items(), key=lambda x: x[0]))
            all_data["eval_times"] = dict(sorted(all_data["eval_times"].items(), key=lambda x: x[0]))
            isSorted = True
        except:""
    return all_data['results'], all_data["train_times"], all_data["eval_times"]


#############################################
# Create graphs
#############################################   

import matplotlib.pyplot as plt

def create_scatter_line_plot(data, title, xlabel, ylabel, y_min = None, y_max = None):
    """Create a scatter graph with a line passing through the mean values

    Args:
        data (dict): Data used to create the graph. The keys are the x values and the values are the y values
        title (string): Title of the graph
        xlabel (string): Name of the x axis
        ylabel (string): Name of the y axis
    """
    resultsMeans = {}

    for key in data.keys():
        if len(data[key]) > 0:
            resultsMeans[key] = np.mean(data[key])
        else:
            resultsMeans[key] = 0.0

    xMean = list(resultsMeans.keys())
    yMean = list(resultsMeans.values())

    listOfLists = list(data.values())

    xAll = []
    for i in range(len(listOfLists)): # for each key
        for _ in range(len(listOfLists[i])): # for each repetition of the key
            xAll.append(xMean[i])
    yAll = np.concatenate(list(data.values()))
    
    if len(data) > 0:
        if(y_min is None):
            y_min = min(yAll)
        if(y_max is None):
            y_max = max(yAll)
    
    plt.figure(figsize=(8, 6))
    plt.ylim(y_min, y_max)
    plt.plot(xMean, yMean, marker='', linestyle='-')
    plt.scatter(xAll, yAll)
    
    for i in range(len(xMean)):
        if (y_max is None or yMean[i] <= y_max) and (y_min is None or yMean[i] >= y_min):
            plt.text(xMean[i], yMean[i], f'{yMean[i]:.2f}', ha='center', bbox = dict(facecolor = 'white', alpha =.8))
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.grid(True)
    plt.show()

def create_bar_plot(data, title, xlabel, ylabel, vertical_xticks=False, custom_xticks=None, y_min = None, y_max = None):
    """Create a bar graph

    Args:
        data (dict): Data used to create the graph. The keys are the x values and the values are the y values
        title (string): Title of the graph
        xlabel (string): Name of the x axis
        ylabel (string): Name of the y axis
        vertical_xticks (bool, optional): If True then the x values are vertical otherwise they are horizontal (useful for long text labels). Defaults to False.
        custom_xticks (list, optional): List of values for the x axis. If None then the keys in data are used. Defaults to None.
    """    
    resultsMeans = {}

    for key in data.keys():
        if len(data[key]) > 0:
            resultsMeans[key] = np.mean(data[key])

    xMean = list(resultsMeans.keys())
    yMean = list(resultsMeans.values())
    
    if len(data) > 0:
        if y_min is None:
            y_min = 0
        if y_max is None:
            y_max = max(yMean)
    
    plt.figure(figsize=(11, 6))
    if vertical_xticks:
        plt.xticks(fontsize=15, rotation='vertical')
    else:
        plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    if not (custom_xticks is None):
        if type(custom_xticks) == type([]) and len(custom_xticks) == len(xMean):
            xMean = custom_xticks
        else:
            print("WARNING: The array of x ticks must be of the same length of the one created without it (when custom_xticks=None). The xticks were not changed to the given ones.")

    for i in range(len(yMean)):
        if (y_max is None or yMean[i] <= y_max) and (y_min is None or yMean[i] >= y_min):
            plt.text(i, yMean[i], f'{yMean[i]:.2f}', ha = 'center', bbox = dict(facecolor = 'white', alpha =.8))
 
    
    plt.ylim(y_min, y_max)
    plt.bar(xMean, yMean)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()
 
 
def create_boxplot(data, title, xlabel, ylabel, vertical_xticks=False, custom_xticks=None, y_min = None, y_max = None):
    """Create a box plot

    Args:
        data (dict): Data used to create the graph. The keys are the x values and the values are the y values
        title (string): Title of the graph
        xlabel (string): Name of the x axis
        ylabel (string): Name of the y axis
        vertical_xticks (bool, optional): If True then the x values are vertical otherwise they are horizontal (useful for long text labels). Defaults to False.
        custom_xticks (list, optional): List of values for the x axis. If None then the keys in data are used. Defaults to None.
    """
    
    medians = {}
    minY = float('+inf')
    maxY = float('-inf')
    for key, value in data.items():
        medians[key] = np.median(value)
        if len(value) > 0:
            value_min = min(value)
            value_max = max(value)
            if value_min < minY:
                minY = value_min
            elif value_max > maxY:
                maxY = value_max
        
    if len(data) > 0:
        if y_min is None and minY != float('+inf'):
            y_min = minY
        if y_max is None and maxY != float('-inf'):
            y_max = maxY
    
    plt.figure(figsize=(8, 6))
    plt.ylim(y_min, y_max)
    plt.boxplot(data.values())
    
    labels = data.keys()
    
    if not (custom_xticks is None):
        if type(custom_xticks) == type([]) and len(custom_xticks) == len(labels):
            labels = custom_xticks
        else:
            print("WARNING: The array of x ticks must be of the same length of the one created without it (when custom_xticks=None). The xticks were not changed to the given ones.")
    
    if vertical_xticks:
        plt.xticks(ticks=list(range(1,len(data)+1)) ,labels=labels, rotation='vertical')
    else:
        plt.xticks(ticks=list(range(1,len(data)+1)) ,labels=labels)
    
    i = 1 
    for key in data.keys():
        if (y_max is None or medians[key] <= y_max) and (y_min is None or medians[key] >= y_min):
            plt.text(i, medians[key], f'{medians[key]:.2f}', ha='center', bbox = dict(facecolor = 'white', alpha =.8))
        i += 1
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.grid(True)
    plt.show()
    

#############################################
# Split dataset (training & test sets)
#############################################    
    
# Here we don't use sklearn's function since we just want to split the dataset and we are not separating labels and descriptive variables
def split_dataset(dataset, ratio):
    """Split a dataset in two parts

    Args:
        dataset (pandas.DataFrame)
        ratio (integer): Ratio of the size of the first subset compared to the whole dataset

    Returns:
        (pandas.DataFrame): First subset (size = dataset size * ratio)
        (pandas.DataFrame): Second subset (size = dataset size * (1-ratio))
    """
    first_set = dataset.sample(frac = ratio, random_state=42)
    second_set = dataset.drop(first_set.index)
    return first_set, second_set

#############################################
# Create support set (n samples per class)
#############################################   

def get_n_shot_dataset(dataset, n_samples_per_class):
    new_dataset = dataset.groupby('label').head(n_samples_per_class)
    return new_dataset