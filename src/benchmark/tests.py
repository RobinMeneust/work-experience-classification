import torch
from setfit import sample_dataset
import gc
import time
import pandas as pd
import numpy as np
from benchmark.utility import get_remaining_time_str
from datasets import Dataset
import multiprocessing

#############################################
# Apply length or language filters
#############################################

from langdetect import detect

def filter_lang(data, lang):
    """Only keep the rows in which the text column is in the given language

    Args:
        data (datasets.Dataset): Data to be filtered
        lang (string): Language used to filter the data (e.g. 'fr' or 'en')

    Returns:
        datasets.Dataset: Filtered data
    """
    indices = []
    for i in range(len(data)):
        try:
            l = detect(data.iloc[i]["text"])
            if l == lang:
                indices.append(i)
        except:""
    
    return data.iloc[indices]

def filter_dataset(data, min_text_length=None, max_text_length=None, lang=None):
    """Filter a dataset on language and length (in number of words) filters

    Args:
        data (datasets.Dataset): Dataset to be filtered
        min_text_length (int, optional): Min length of the string in the text field. If the length is lesser than it then the row is dropped. If it's None there is no filter. Defaults to None.
        max_text_length (int, optional): Max length of the string in the text field. If the length is greater than it then the row is dropped. If it's None there is no filter. Defaults to None.
        lang (string, optional): Language used to filter the data (e.g. 'fr' or 'en'). If it's None there is no filter. Defaults to None.

    Returns:
        datasets.Dataset: Filtered dataset
    """
    if min_text_length is None:
        if max_text_length is None:
            filtered_data = data
        else:
            filtered_data = data[data['text'].str.split().apply(len) <= max_text_length]
    else:
        if max_text_length is None:
            filtered_data = data[data['text'].str.split().apply(len) >= min_text_length]
        else:
            filtered_data = data[data['text'].str.split().apply(len).between(min_text_length, max_text_length)]
            
    if not (lang is None):
        filtered_data = filter_lang(filtered_data, lang)
           
    return filtered_data


#############################################
# N-shots test
#############################################


def n_shot_tests(params, train_set, test_set, few_shot_model_f1_function):
    """Run a SetFit N-shots test

    Args:
        params (dict): Test parameters (number of iterations...)
        train_set (datasets.Dataset): Training set
        test_set (datasets.Dataset): Test set

    Returns:
        results (dict): Test results (F1-scores) for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of F1-scores)
        run_times (dict): Run times for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of the training run times)
    """
    
    n_values = params["n_shot"]
    n_iter = params["n_iter"]
    n_max_iter_per_shot = params["n_max_iter_per_shot"]
    model = params["model"]
    loss = params["loss"]
    input_length_range = params["input_length_range"] if "input_length_range" in params else None
    num_epochs = params["num_epochs"] if "num_epochs" in params else None
    batch_size = params["batch_size"] if "batch_size" in params else None
    ratio_frozen_weights = params["ratio_frozen_weights"] if "ratio_frozen_weights" in params else None
    
    n_values_max = np.max(n_values)

    results = {}
    run_times = {}
    for n_shot in n_values:
        results[n_shot] = []
        run_times[n_shot] = []
    
    progress = 0
    progress_end = n_iter * ((len(n_values)-1)*n_max_iter_per_shot + 1)
    
    start_time = time.time()
    # Repeat the tests multiple times because F1-score variations might be due to the examples chosen and not the input length of those examples
    for i in range(n_iter):
        # Use the same subset of the dataset for all of the tests in the following loop
        if not (input_length_range is None):
            new_train_set = filter_dataset(train_set, input_length_range[0], input_length_range[1])
        else:
            new_train_set = train_set
        new_train_set = new_train_set.sample(frac = 1, random_state=i*47).groupby('label').head(n_values_max)
        new_train_set = Dataset.from_pandas(new_train_set, split="train")
        new_test_set = Dataset.from_pandas(test_set, split="test")

        for n_shot in n_values:
            i_shot = 0
            try:
                n_iter_shot = n_max_iter_per_shot if n_shot < n_values_max else 1
                for i_shot in range(n_iter_shot):
                    progress += 1
                    print("Step:", progress, "/", progress_end,"Estimated remaining time:", get_remaining_time_str(start_time, progress, progress_end))
        
                    train_set_n_shot = sample_dataset(new_train_set, label_column="label", num_samples=n_shot, seed=47*n_shot + 3*i_shot)
                    f1_score, run_time = run_test_job(target=few_shot_model_f1_function, kwargs={"train_set":train_set_n_shot, "test_set":new_test_set, "model_name":model, "loss":loss, "num_epochs":num_epochs, "batch_size":batch_size, "ratio_frozen_weights":ratio_frozen_weights})
                    results[n_shot].append(f1_score)
                    run_times[n_shot].append(run_time)
            except Exception as err:
                print(n_shot, "failed", str(err))
                progress += n_iter_shot - i_shot - 1
    return results, run_times


#############################################
# Input length test
#############################################

def input_length_tests(params, train_set, test_set, few_shot_model_f1_function):
    """Run a SetFit input length test

    Args:
        params (dict): Test parameters (number of iterations...)
        train_set (datasets.Dataset): Training set
        test_set (datasets.Dataset): Test set

    Returns:
        results (dict): Test results (F1-scores) for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of F1-scores)
        run_times (dict): Run times for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of the training run times)
    """
    
    n_shot = params["n_shot"]
    len_values = params["input_length_range"]
    n_iter = params["n_iter"]
    model = params["model"]
    loss = params["loss"]
    num_epochs = params["num_epochs"] if "num_epochs" in params else None
    batch_size = params["batch_size"] if "batch_size" in params else None
    ratio_frozen_weights = params["ratio_frozen_weights"] if "ratio_frozen_weights" in params else None
    
    results = {}
    run_times = {}
    new_test_set = Dataset.from_pandas(test_set, split="test")

    for i in range(len(len_values)):
        key = f"[{len_values[i][0]},{len_values[i][1]}]"
        results[key] = []
        run_times[key] = []

    progress = 0
    progress_end = n_iter * len(len_values)
    start_time = time.time()
    
    # Repeat the tests multiple times because F1-score variations might be due to the examples chosen and not the input length of those examples
    for iter in range(n_iter):
        for i in range(len(len_values)):
            key = f"[{len_values[i][0]},{len_values[i][1]}]"
            try:
                progress += 1
                print("Step:", progress, "/", progress_end,"Estimated remaining time:", get_remaining_time_str(start_time, progress, progress_end))
    
                new_train_set = filter_dataset(train_set, len_values[i][0], len_values[i][1])
                new_train_set = Dataset.from_pandas(new_train_set, split="train")
                new_train_set = sample_dataset(new_train_set, label_column="label", num_samples=n_shot, seed=47*iter)
                f1_score, run_time = run_test_job(target=few_shot_model_f1_function, kwargs={"train_set":new_train_set, "test_set":new_test_set, "model_name":model, "loss":loss, "num_epochs":num_epochs, "batch_size":batch_size, "ratio_frozen_weights":ratio_frozen_weights})
                results[key].append(f1_score)
                run_times[key].append(run_time)
            except Exception as err:
                print(key, "failed", str(err))
    return results, run_times


#############################################
# Distance metric test
#############################################


def distance_tests(params, train_set, test_set, few_shot_model_f1_function):
    """Run a SetFit distance test

    Args:
        params (dict): Test parameters (number of iterations...)
        train_set (datasets.Dataset): Training set
        test_set (datasets.Dataset): Test set

    Returns:
        results (dict): Test results (F1-scores) for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of F1-scores)
        run_times (dict): Run times for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of the training run times)
    """
    
    n_shot = params["n_shot"]
    n_iter = params["n_iter"]
    model = params["model"]
    loss = params["loss"]
    distances = params["distance"]
    input_length_range = params["input_length_range"] if "input_length_range" in params else None
    num_epochs = params["num_epochs"] if "num_epochs" in params else None
    batch_size = params["batch_size"] if "batch_size" in params else None
    ratio_frozen_weights = params["ratio_frozen_weights"] if "ratio_frozen_weights" in params else None

    results = {}
    run_times = {}
    for key in distances.keys():
        results[key] = []
        run_times[key] = []

    new_test_set = Dataset.from_pandas(test_set, split="test")
 
    progress = 0
    progress_end = n_iter * (len(distances))
    start_time = time.time()
    
    for i in range(n_iter):
        if not (input_length_range is None):
            new_train_set = filter_dataset(train_set, input_length_range[0], input_length_range[1])
        else:
            new_train_set = train_set
        new_train_set = Dataset.from_pandas(new_train_set, split="train")
        new_train_set = sample_dataset(new_train_set, label_column="label", num_samples=n_shot, seed=47*i)
        
        for key in distances.keys():
            progress += 1
            print("Step:", progress, "/", progress_end,"Estimated remaining time:", get_remaining_time_str(start_time, progress, progress_end))
   
            try:
                f1_score, run_time = run_test_job(target=few_shot_model_f1_function, kwargs={"train_set":new_train_set, "test_set":new_test_set, "model_name":model, "loss":loss, "distance_metric":distances[key], "num_epochs":num_epochs, "batch_size":batch_size, "ratio_frozen_weights":ratio_frozen_weights})
                results[key].append(f1_score)
                run_times[key].append(run_time)
            except Exception as err:
                print(key, "failed", str(err))
    return results, run_times


#############################################
# Loss test
#############################################

def loss_tests(params, train_set, test_set, few_shot_model_f1_function):
    """Run a SetFit loss test

    Args:
        params (dict): Test parameters (number of iterations...)
        train_set (datasets.Dataset): Training set
        test_set (datasets.Dataset): Test set

    Returns:
        results (dict): Test results (F1-scores) for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of F1-scores)
        run_times (dict): Run times for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of the training run times)
    """
    n_shot = params["n_shot"]
    n_iter = params["n_iter"]
    model = params["model"]
    losses = params["loss"]
    input_length_range = params["input_length_range"] if "input_length_range" in params else None
    num_epochs = params["num_epochs"] if "num_epochs" in params else None
    batch_size = params["batch_size"] if "batch_size" in params else None
    ratio_frozen_weights = params["ratio_frozen_weights"] if "ratio_frozen_weights" in params else None
    
    results = {}
    run_times = {}
    for key in losses.keys():
        results[key] = []
        run_times[key] = []

    new_test_set = Dataset.from_pandas(test_set, split="test")
 
    progress = 0
    progress_end = n_iter * (len(losses))
    start_time = time.time()
    
    for i in range(n_iter):
        if not (input_length_range is None):
            new_train_set = filter_dataset(train_set, input_length_range[0], input_length_range[1])
        else:
            new_train_set = train_set
        new_train_set = Dataset.from_pandas(new_train_set, split="train")
        new_train_set = sample_dataset(new_train_set, label_column="label", num_samples=n_shot, seed=47*i)
        
        for key in losses.keys():
            progress += 1
            print("Step:", progress, "/", progress_end,"Estimated remaining time:", get_remaining_time_str(start_time, progress, progress_end))
   
            try:
                f1_score, run_time = run_test_job(target=few_shot_model_f1_function, kwargs={"train_set":new_train_set, "test_set":new_test_set, "model_name":model, "loss":losses[key], "num_epochs":num_epochs, "batch_size":batch_size, "ratio_frozen_weights":ratio_frozen_weights})
                results[key].append(f1_score)
                run_times[key].append(run_time)
            except Exception as err:
                print(key, "failed", str(err))
    return results, run_times

#############################################
# Language test
#############################################

def language_tests(params, train_set, test_set, few_shot_model_f1_function):
    """Run a SetFit language test

    Args:
        params (dict): Test parameters (number of iterations...)
        train_set (datasets.Dataset): Training set
        test_set (datasets.Dataset): Test set

    Returns:
        results (dict): Test results (F1-scores) for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of F1-scores)
        run_times (dict): Run times for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of the training run times)
    """
    
    n_shot = params["n_shot"]
    n_iter = params["n_iter"]
    model = params["model"]
    loss = params["loss"]
    languages = params["lang"]
    input_length_range = params["input_length_range"] if "input_length_range" in params else None
    num_epochs = params["num_epochs"] if "num_epochs" in params else None
    batch_size = params["batch_size"] if "batch_size" in params else None
    ratio_frozen_weights = params["ratio_frozen_weights"] if "ratio_frozen_weights" in params else None
    
    results = {}
    run_times = {}

    for key in languages:
        results[key] = []
        run_times[key] = []
    results['all'] = []
    run_times['all'] = []
 
    progress = 0
    progress_end = n_iter * (len(languages) + 1)
    start_time = time.time()
    
    for i in range(n_iter):
        temp_train_set_panda = {}
        temp_test_set_panda = {}

        for key in languages:
            progress += 1
            print("Step:", progress, "/", progress_end,"Estimated remaining time:", get_remaining_time_str(start_time, progress, progress_end))
   
            temp_train_set_panda[key] = filter_dataset(train_set, lang=key)
            if not (input_length_range is None):
                temp_train_set_panda[key] = filter_dataset(temp_train_set_panda[key], input_length_range[0], input_length_range[1])
            temp_train_set = Dataset.from_pandas(temp_train_set_panda[key], split="train")
            temp_train_set = sample_dataset(temp_train_set, label_column="label", num_samples=n_shot, seed=47*i)
   
            temp_test_set_panda[key] = filter_dataset(test_set, lang=key)
            temp_test_set = Dataset.from_pandas(temp_test_set_panda[key], split="test")
            try:
                f1_score, run_time = run_test_job(target=few_shot_model_f1_function, kwargs={"train_set":temp_train_set, "test_set":temp_test_set, "model_name":model, "loss":loss, "num_epochs":num_epochs, "batch_size":batch_size, "ratio_frozen_weights":ratio_frozen_weights})
                results[key].append(f1_score)
                run_times[key].append(run_time)
            except Exception as err:
                print(key, "failed", str(err))
                del temp_train_set_panda[key]
                del temp_test_set_panda[key]
        
        all_temp_train_set = list(temp_train_set_panda.values())
        all_temp_test_set = list(temp_train_set_panda.values())
  
        if len(all_temp_train_set) == 0 or len(all_temp_test_set) == 0:
            progress += 1
            print("Step:", progress, "/", progress_end, "failed")
            continue

        progress += 1
        print("Step:", progress, "/", progress_end,"Estimated remaining time:", get_remaining_time_str(start_time, progress, progress_end))
        
        try:
            all_train_set = pd.concat(all_temp_train_set)
            all_train_set = Dataset.from_pandas(all_train_set, split="test")
            all_train_set = sample_dataset(all_train_set, label_column="label", num_samples=n_shot, seed=47*i)
            all_test_set = pd.concat(all_temp_test_set)
            all_test_set = Dataset.from_pandas(all_test_set, split="test")
  
            f1_score, run_time = run_test_job(target=few_shot_model_f1_function, kwargs={"train_set":all_train_set, "test_set":all_test_set, "model_name":model, "loss":loss, "num_epochs":num_epochs, "batch_size":batch_size, "ratio_frozen_weights":ratio_frozen_weights})
            results['all'].append(f1_score)
            run_times['all'].append(run_time)
        except Exception as err:
            print('all', "failed", str(err))
    return results, run_times


#############################################
# Sentence Transformer models test
#############################################

def model_tests(params, train_set, test_set, few_shot_model_f1_function):
    """Run a SetFit model test

    Args:
        params (dict): Test parameters (number of iterations...)
        train_set (datasets.Dataset): Training set
        test_set (datasets.Dataset): Test set

    Returns:
        results (dict): Test results (F1-scores) for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of F1-scores)
        run_times (dict): Run times for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of the training run times)
    """
    
    n_shot = params["n_shot"]
    n_iter = params["n_iter"]
    loss = params["loss"]
    models = params["model"]
    input_length_range = params["input_length_range"] if "input_length_range" in params else None
    num_epochs = params["num_epochs"] if "num_epochs" in params else None
    batch_size = params["batch_size"] if "batch_size" in params else None
    ratio_frozen_weights = params["ratio_frozen_weights"] if "ratio_frozen_weights" in params else None
    
    results = {}
    run_times = {}
    new_test_set = Dataset.from_pandas(test_set, split="test")
    start_time = time.time()

    for key in models.keys():
        results[key] = []
        run_times[key] = []

    progress = 0
    progress_end = n_iter * len(models)
 
    for i in range(n_iter):
        # Use the same subset of the dataset for all of the tests in the following loop
        if not (input_length_range is None):
            new_train_set = filter_dataset(train_set, input_length_range[0], input_length_range[1])
        else:
            new_train_set = train_set
        new_train_set = Dataset.from_pandas(new_train_set, split="train")
        new_train_set = sample_dataset(new_train_set, label_column="label", num_samples=n_shot, seed=47*i)

        for key, full_model_name in models.items():
            progress += 1
            print("Step:", progress, "/", progress_end,"Estimated remaining time:", get_remaining_time_str(start_time, progress, progress_end))
   
            try:
                f1_score, run_time = run_test_job(target=few_shot_model_f1_function, kwargs={"train_set":new_train_set, "test_set":new_test_set, "model_name":full_model_name, "loss":loss, "num_epochs":num_epochs, "batch_size":batch_size, "ratio_frozen_weights":ratio_frozen_weights})
                results[key].append(f1_score)
                run_times[key].append(run_time)
            except Exception as err:
                print(key, "failed", str(err))
    return results, run_times

#############################################
# Number of epochs test
#############################################

def num_epochs_tests(params, train_set, test_set, few_shot_model_f1_function):
    """Run a SetFit test on the number of epochs

    Args:
        params (dict): Test parameters (number of iterations...)
        train_set (datasets.Dataset): Training set
        test_set (datasets.Dataset): Test set

    Returns:
        results (dict): Test results (F1-scores) for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of F1-scores)
        run_times (dict): Run times for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of the training run times)
    """
    n_shot = params["n_shot"]
    n_iter = params["n_iter"]
    loss = params["loss"]
    model = params["model"]
    input_length_range = params["input_length_range"] if "input_length_range" in params else None
    num_epochs = params["num_epochs"]
    batch_size = params["batch_size"] if "batch_size" in params else None
    ratio_frozen_weights = params["ratio_frozen_weights"] if "ratio_frozen_weights" in params else None
    
    results = {}
    run_times = {}
    new_test_set = Dataset.from_pandas(test_set, split="test")

    for epoch_tuple in num_epochs:
        key = f"({epoch_tuple[0]}, {epoch_tuple[1]})"
        results[key] = []
        run_times[key] = []

    progress = 0
    progress_end = n_iter * len(num_epochs)
    start_time = time.time()
    
    for i in range(n_iter):
        # Use the same subset of the dataset for all of the tests in the following loop
        if not (input_length_range is None):
            new_train_set = filter_dataset(train_set, input_length_range[0], input_length_range[1])
        else:
            new_train_set = train_set
        new_train_set = Dataset.from_pandas(new_train_set, split="train")
        new_train_set = sample_dataset(new_train_set, label_column="label", num_samples=n_shot, seed=47*i)

        for epoch_tuple in num_epochs:
            key = f"({epoch_tuple[0]}, {epoch_tuple[1]})"
            progress += 1
            print("Step:", progress, "/", progress_end,"Estimated remaining time:", get_remaining_time_str(start_time, progress, progress_end))
   
            try:
                f1_score, run_time = run_test_job(target=few_shot_model_f1_function, kwargs={"train_set":new_train_set, "test_set":new_test_set, "model_name":model, "loss":loss, "num_epochs":epoch_tuple, "batch_size":batch_size, "ratio_frozen_weights":ratio_frozen_weights})
                results[key].append(f1_score)
                run_times[key].append(run_time)
            except Exception as err:
                print(key, "failed", str(err))
    return results, run_times


#############################################
# Data sampling test (constant parameters)
#############################################

def constant_params_tests(params, train_set, test_set, few_shot_model_f1_function):
    """Run a SetFit test with constant parameters (but we still iterate multiple times to test with different training sets)

    Args:
        params (dict): Test parameters (number of iterations...)
        train_set (datasets.Dataset): Training set
        test_set (datasets.Dataset): Test set

    Returns:
        results (dict): Test results (F1-scores) for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of F1-scores)
        run_times (dict): Run times for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of the training run times)
    """
    n_shot = params["n_shot"]
    n_iter = params["n_iter"]
    loss = params["loss"]
    model = params["model"]
    num_epochs = params["num_epochs"] if "num_epochs" in params else None
    batch_size = params["batch_size"] if "batch_size" in params else None
    input_length_range = params["input_length_range"] if "input_length_range" in params else None
    ratio_frozen_weights = params["ratio_frozen_weights"] if "ratio_frozen_weights" in params else None
    
    results = []
    run_times = []
    new_test_set = Dataset.from_pandas(test_set, split="test")
    progress = 0

    for i in range(n_iter):
        progress += 1
        print("Step:", progress, "/", n_iter)
        if not (input_length_range is None):
            new_train_set = filter_dataset(train_set, input_length_range[0], input_length_range[1])
        else:
            new_train_set = train_set
        new_train_set = Dataset.from_pandas(new_train_set, split="train")
        new_train_set = sample_dataset(new_train_set, label_column="label", num_samples=n_shot, seed=47*i)

        try:
            f1_score, run_time = run_test_job(target=few_shot_model_f1_function, kwargs={"train_set":new_train_set, "test_set":new_test_set, "model_name":model, "loss":loss, "num_epochs":num_epochs, "batch_size":batch_size, "ratio_frozen_weights":ratio_frozen_weights})
            results.append(f1_score)
            run_times.append(run_time)
        except Exception as err:
                print(i, "failed", str(err))
    return {"all":results}, {"all":run_times}


#############################################
# Data augmentation tests
#############################################


# Synonym replacement

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
import random

nltk.download('wordnet')

def to_wordnet_pos(pos):
    if pos.startswith('J'):
        return wordnet.ADJ
    elif pos.startswith('V'):
        return wordnet.VERB
    elif pos.startswith('N'):
        return wordnet.NOUN
    elif pos.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def get_synonyms(word, pos, lang):
    """Get the synonyms of a word in the given language using WordNet

    Args:
        word (string): Word whose synonyms are searched
        pos (string): Part of speech tag
        lang (string): Language of the word (e.g. 'fr' or 'en')

    Returns:
        list: List of synonyms (strings)
    """
    l = 'fra' if lang == 'fr' else 'eng'
    synonyms=[synset.lemma_names(l) for synset in wordnet.synsets(word, pos=to_wordnet_pos(pos), lang=l)]
    
    synonyms = sum(synonyms, []) # concatenate the nested list synonyms
    synonyms = list(set(synonyms)) # remove duplicates
    
    if word in synonyms:
        synonyms.remove(word)
    
    return synonyms

def replace_with_synonym(word, pos, lang):
    """Replace a word with one of its synonym (one is chosen randomly)

    Args:
        word (string): Word replaced
        pos (string): Part of speech tag
        lang (string): Language of the word (e.g. 'fr' or 'en')

    Returns:
        string: New word
    """
    synonyms = get_synonyms(word, pos, lang)
    if synonyms:
        syn = random.choice(synonyms)
        word = syn
    return word

def apply_synonym_replacement_to_text(text, lang, params=None):
    """Replace random words with random synonyms. The probability of replacement can be defined in the params object

    Args:
        text (string): Text to be modified with synonyms
        lang (string): lang (string): Language of the word (e.g. 'fr' or 'en')
        params (dict, optional): Parameters of the replacement (e.g. modification_rate gives the probability for each word to be replaced). Defaults to None.

    Raises:
        Exception: If the parameters are invalid (e.g. modification_rate (float) not in [0,1])

    Returns:
        string: Text modified with synonyms
    """
    modification_rate = 1
    if not (params is None) and "modification_rate" in params :
        if params["modification_rate"] < 0 or params["modification_rate"] > 1:
            raise Exception("Invalid modification_rate (expected a value between 0 and 1)")
        else:
            modification_rate = params["modification_rate"]
        
    sentences = sent_tokenize(text)
    augmented_sentences = []

    opposite_modification_ratio = 1 - modification_rate	
 
    for sentence in sentences:
        tokenized = word_tokenize(sentence)
        pos_tags = nltk.pos_tag(tokenized)
        augmented_tokens = []
        for token, pos in pos_tags:
            do_replace_by_synonym = 1 == np.random.choice([0,1], p=[opposite_modification_ratio,modification_rate])
            if do_replace_by_synonym:
                augmented_tokens.append(replace_with_synonym(token, pos, lang))
            else:
                augmented_tokens.append(token)
        augmented_sentence = ' '.join(augmented_tokens)
        augmented_sentences.append(augmented_sentence)
     
    augmented_text = ' '.join(augmented_sentences)
    return augmented_text

def augment_synonym_replacement(data, n_new_samples_per_class, classes, strategy_params):
    """Augment a dataset with new samples using synonym replacement

    Args:
        data (datasets.Dataset): Dataset to be augmented
        n_new_samples_per_class (int): Number of new samples to be generated
        classes (list): List of class labels (e.g [0,1])
        strategy_params (dict): Parameters of the replacement (e.g. modification_rate gives the probability for each word to be replaced).

    Returns:
       dict : New generated samples. The key is the class labels and the value is the new text
    """
    progress = 0
    progress_end = n_new_samples_per_class * len(classes)
    do_continue = True
    
    new_samples = {}
    for c in classes:
        new_samples[c] = []
    
    while(do_continue and progress < progress_end):     
        for i in range(len(data)):
            c = data.iloc[i]["label"]
            if(len(new_samples[c]) >= n_new_samples_per_class):
                continue
                
            print("Data augmentation... (", progress, "/", progress_end,")")
            try:
                l = detect(data.iloc[i]["text"])
                if l != 'fr' and l != 'en':
                    continue
                new_samples[c].append(apply_synonym_replacement_to_text(data.iloc[i]["text"], l, strategy_params))
            except Exception as err:
                print("failed", str(err))
    
            progress += 1
            if progress >= progress_end:
                break

        do_continue = False
        for val in new_samples.values():
            if len(val) < n_new_samples_per_class:
                do_continue = True
    
    return new_samples

# Swap all sentences

from random import randint

def gen_random_text_from_sentences(sentences, length):
    """Create a text of a given length from a list of sentences

    Args:
        sentences (list): List of sentences (strings) that are combined
        length (_type_): _description_

    Returns:
        _type_: _description_
    """
    selected_sentences = []
    for _ in range(min(length,len(sentences))):
        selected = random.choice(sentences)
        selected_sentences.append(selected)
        sentences.remove(selected)
    return ' '.join(selected_sentences)
     

def augment_swapping_inter(data, n_new_samples_per_class, classes):
    """Augment a dataset with new samples using swapping all samples from the same class
    All the samples sentences are extracted and re-merged randomly (with a random number of sentences)

    Args:
        data (datasets.Dataset): Dataset to be augmented
        n_new_samples_per_class (int): Number of new samples to be generated
        classes (list): List of class labels (e.g [0,1])

    Returns:
       dict : New generated samples. The key is the class labels and the value is the new text
    """
    progress = 0
    progress_end = n_new_samples_per_class * len(classes)
    max_text_size = 5 # max number of sentences. #TODO: Add to the params of this function
    
    new_samples = {}
    for c in classes:
        new_samples[c] = []
        filtered_rows = data[data['label'] == c]
        sentences = []
        for i in range(len(filtered_rows)):
            sentences = sentences + sent_tokenize(filtered_rows.iloc[i]["text"])

        while(len(new_samples[c]) < n_new_samples_per_class):
            progress += 1
            print("Data augmentation... (", progress, "/", progress_end,")")
   
            length = randint(1, max(1,min(len(sentences)//4, max_text_size)))
            new_text = gen_random_text_from_sentences(sentences, length)
            new_samples[c].append(new_text)
                
    return new_samples

# Crossover

from math import floor

def crossover(sentences_parent1, sentences_parent2, n_sections):
    """Crossover of 2 texts

    To generate a new sample, two are taken randomly and are then cut in N points. Finally the swap the different sections between the two samples.
    Example (1 point crossover = 2 sections):
        - sample 1:     S1. S2. S3. S4
        - sample 2:     S5. S6. S7. S8
        - new sample:   S1. S2. S7. S8
        
    Args:
        sentences_parent1 (list): List of sentences (strings) of the parent 1
        sentences_parent2 (list): List of sentences (strings) of the parent 2
        n_sections (int): Number of sections (e.g if it's 3 then the text of parent one (and 2) is split in 3)
    """
    chunk_len_parent1 = len(sentences_parent1) // n_sections + (1 if len(sentences_parent1) % n_sections != 0 else 0)
    chunk_len_parent2 = len(sentences_parent2) // n_sections + (1 if len(sentences_parent2) % n_sections != 0 else 0)
    i1 = 0
    i2 = 0
    
    augmented_sentences = []

    i_parent = random.choice([0,1])
    for s in range(n_sections):
        if i1>=len(sentences_parent1):
            if i2<len(sentences_parent2):
                while i2 < len(sentences_parent2):
                    augmented_sentences.append(sentences_parent2[i2])
                    i2 += 1
        elif i2>=len(sentences_parent2):
            while i2 < len(sentences_parent2):
                augmented_sentences.append(sentences_parent1[i1])
                i1 += 1
        else:
            if i_parent == 0:
                i_chunk = 0
                while i_chunk < chunk_len_parent1 and i1+i_chunk<len(sentences_parent1):
                    augmented_sentences.append(sentences_parent1[i1 + i_chunk])
                    i_chunk += 1
            else:
                i_chunk = 0
                while i_chunk < chunk_len_parent2 and i2+i_chunk<len(sentences_parent2):
                    augmented_sentences.append(sentences_parent2[i2 + i_chunk])
                    i_chunk += 1
            i1 += chunk_len_parent1
            i2 += chunk_len_parent2
            i_parent = 1-i_parent
    
    return augmented_sentences

def augment_crossover(data, n_new_samples_per_class, classes, strategy_params=None):
    """Augment a dataset with new samples using crossover

    Args:
        data (datasets.Dataset): Dataset to be augmented
        n_new_samples_per_class (int): Number of new samples to be generated
        classes (list): List of class labels (e.g [0,1])
        strategy_params (dict): Parameters (e.g. n_points_crossover gives the number of points were we "cut" the samples).

    Returns:
       dict : New generated samples. The key is the class labels and the value is the new text
    """
    n_points_crossover = floor(strategy_params["n_points_crossover"]) if not (strategy_params is None) and "n_points_crossover" in strategy_params else 1
    if n_points_crossover<0:
        n_points_crossover = 1
    progress = 0
    progress_end = n_new_samples_per_class * len(classes)
    n_sections = n_points_crossover+1
 
    new_samples = {}
    for c in classes:
        new_samples[c] = []
        filtered_rows = data[data['label'] == c]
        
        while(len(new_samples[c]) < n_new_samples_per_class):
            for parent1 in range(len(filtered_rows)):
                progress += 1
                print("Data augmentation... (", progress, "/", progress_end,")")
    
                parent2 =  random.choice([j for j in range(len(filtered_rows)) if j != parent1])
    
                sentences_parent1 = sent_tokenize(filtered_rows.iloc[parent1]["text"])
                sentences_parent2 = sent_tokenize(filtered_rows.iloc[parent2]["text"])
                
                augmented_sentences = crossover(sentences_parent1, sentences_parent2, n_sections)               

                new_text = ' '.join(augmented_sentences)
                new_samples[c].append(new_text)

                if len(new_samples[c]) >= n_new_samples_per_class:
                    break
                
    return new_samples


# Back translation

from transformers import MarianMTModel, MarianTokenizer

def back_translate(text, model_fr_en, tokenizer_fr_en, model_en_fr, tokenizer_en_fr, num_generations):
    """Back translate a text (fr -> en -> fr or en -> fr -> en)

    Args:
        text (string): Text to be back translated
        model_fr_en (MarianMTModel): AI model to translate french tokens to english tokens
        tokenizer_fr_en (MarianTokenizer): French tokenizer
        model_en_fr (MarianMTModel): AI model to translate english tokens to french tokens
        tokenizer_en_fr (MarianTokenizer): English tokenizer

    Raises:
        Exception: If the language could not be detected

    Returns:
        list: List of texts back translated from the given one
    """
    l = detect(text)
    if l == 'fr':
        temp = model_fr_en.generate(**tokenizer_fr_en(">>en<< "+text, return_tensors="pt", padding=True))[0]
        temp = tokenizer_fr_en.decode(temp, skip_special_tokens=True)
        temp = model_en_fr.generate(**tokenizer_en_fr(">>fr<< "+temp, return_tensors="pt", padding=True), num_beams=num_generations, num_return_sequences=num_generations)
        return [tokenizer_en_fr.decode(temp[i], skip_special_tokens=True) for i in range(len(temp))]
    elif l == 'en':
        temp = model_en_fr.generate(**tokenizer_en_fr(">>fr<< "+text, return_tensors="pt", padding=True))[0]
        temp = tokenizer_en_fr.decode(temp, skip_special_tokens=True)
        temp = model_fr_en.generate(**tokenizer_fr_en(">>en<< "+temp, return_tensors="pt", padding=True), num_beams=num_generations, num_return_sequences=num_generations)
        return [tokenizer_fr_en.decode(temp[i], skip_special_tokens=True) for i in range(len(temp))]
    else:
        raise Exception("The text is neither FR nor EN")


def augment_back_translation(data, n_new_samples_per_class, classes):
    """Augment a dataset with new samples using back translation

    Args:
        data (datasets.Dataset): Dataset to be augmented
        n_new_samples_per_class (int): Number of new samples to be generated
        classes (list): List of class labels (e.g [0,1])

    Returns:
       dict : New generated samples. The key is the class labels and the value is the new text
    """
    model_name_fr_en = "Helsinki-NLP/opus-mt-fr-en"
    model_name_en_fr = "Helsinki-NLP/opus-mt-en-fr"

    tokenizer_fr_en = MarianTokenizer.from_pretrained(model_name_fr_en)
    tokenizer_en_fr = MarianTokenizer.from_pretrained(model_name_en_fr)
    model_fr_en = MarianMTModel.from_pretrained(model_name_fr_en)
    model_en_fr = MarianMTModel.from_pretrained(model_name_en_fr)
    
    progress = 0
    progress_end = n_new_samples_per_class * len(classes)
    
    new_samples = {}
    for c in classes:
        new_samples[c] = []
    
    num_generations = (n_new_samples_per_class // len(data[data["label"] == 0])) + 1 # When we have less examples than new examples we need to generate several ones from each example
    
    for i in range(len(data)):
        c = data.iloc[i]["label"]
        if(len(new_samples[c]) >= n_new_samples_per_class):
            continue
            
        try:
            new_texts = back_translate(data.iloc[i]["text"], model_fr_en, tokenizer_fr_en, model_en_fr, tokenizer_en_fr, num_generations)
            for t in new_texts:
                if(len(new_samples[c]) >= n_new_samples_per_class):
                    break
                print("Data augmentation... (", progress, "/", progress_end,")")
                new_samples[c].append(t)
        except Exception as err:
            print("failed", str(err))

        progress += 1
        if progress >= progress_end:
            break
    
    del tokenizer_fr_en
    del tokenizer_en_fr
    del model_fr_en
    del model_en_fr
        
    return new_samples

# Main function

# Some methods come from: Li, Bohan, Yutai Hou, and Wanxiang Che. "Data augmentation approaches in natural language processing: A survey." Ai Open 3 (2022): 71-90.

def augment_data(data, n_new_samples_per_class, classes, strategy='synonym', strategy_params = None):
    """Augment a dataset with new samples using the given strategy

    Args:
        data (datasets.Dataset): Dataset to be augmented
        n_new_samples_per_class (int): Number of new samples to be generated
        classes (list): List of class labels (e.g [0,1])
        strategy (string, optional): Name of the data augmentation strategy used. Defaults to 'synonym'.
        strategy_params (dict, optional): Strategy parameters. Defaults to None.

    Raises:
        Exception: If the strategy name is invalid

    Returns:
       datasets.Dataset : Augmented dataset
    """
    
    if n_new_samples_per_class <= 0:
        return data

    if strategy == "swapping_inter":
        new_samples = augment_swapping_inter(data, n_new_samples_per_class, classes)
    elif strategy == "back_translation":
        new_samples = augment_back_translation(data, n_new_samples_per_class, classes)
    elif strategy == "synonym_replacement":
        new_samples = augment_synonym_replacement(data, n_new_samples_per_class, classes, strategy_params)
    elif strategy == "crossover":
        new_samples = augment_crossover(data, n_new_samples_per_class, classes, strategy_params)
    else:
        raise Exception("Unknown strategy")

    gc.collect()
    torch.cuda.empty_cache()

    newData = data.copy()
    for c, samples_list in new_samples.items():
        for sample in samples_list:
            newData.loc[len(newData.index)] = {"text":sample,"label":c}
    return newData

# Function running the tests

def data_augmentation_tests(params, train_set, test_set, few_shot_model_f1_function):
    """Run a SetFit test for data augmentation

    Args:
        params (dict): Test parameters (number of iterations...)
        train_set (datasets.Dataset): Training set
        test_set (datasets.Dataset): Test set

    Returns:
        results (dict): Test results (F1-scores) for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of F1-scores)
        run_times (dict): Run times for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of the training run times)
    """
    
    n_shot = params["n_shot"]
    n_iter = params["n_iter"]
    model = params["model"]
    loss = params["loss"]
    input_length_range = params["input_length_range"] if "input_length_range" in params else None
    augmentation_ratio = params["data_augmentation_ratio"]
    strategy = params["data_augmentation_strategy"]
    num_epochs = params["num_epochs"] if "num_epochs" in params else None
    batch_size = params["batch_size"] if "batch_size" in params else None
    strategy_params = params["strategy_params"] if "strategy_params" in params else None
    ratio_frozen_weights = params["ratio_frozen_weights"] if "ratio_frozen_weights" in params else None
    
    if type(strategy) == type([]) and type(augmentation_ratio) == type([]):
        raise Exception("Only one parameter can be a list of different values, either data_augmentation_strategy or data_augmentation_ratio")

    results = {}
    run_times = {}
    if type(augmentation_ratio) == type([]):
        tested_param_values = augmentation_ratio
        tested_param_key = "data_augmentation_ratio"
        for r in augmentation_ratio:
            results[r] = []
            run_times[r] = []
    else :
        if type(strategy) != type([]):
            strategy = [strategy]
        tested_param_values = strategy
        tested_param_key = "data_augmentation_strategy"
        for s in strategy:
            results[s] = []
            run_times[s] = []
    
    progress = 0
    progress_end = n_iter * len(tested_param_values)
    
    start_time = time.time()
 
    # Repeat the tests multiple times because F1-score variations might be due to the examples chosen and not the input length of those examples
    for i in range(n_iter):
        # Use the same subset of the dataset for all of the tests in the following loop
        if not (input_length_range is None):
            new_train_set = filter_dataset(train_set, input_length_range[0], input_length_range[1])
        else:
            new_train_set = train_set
        new_train_set = new_train_set.sample(frac = 1, random_state=i*47).groupby('label').head(n_shot)
        new_test_set = Dataset.from_pandas(test_set, split="test")

        for val in tested_param_values:
            if tested_param_key == "data_augmentation_ratio":
                n_new_samples_per_class = round((val-1) * n_shot)
                current_strategy = strategy
            else:
                n_new_samples_per_class = round((augmentation_ratio-1) * n_shot)
                current_strategy = val
            try:
                progress += 1
                print("Step:", progress, "/", progress_end,"Estimated remaining time:", get_remaining_time_str(start_time, progress, progress_end))
                
                if n_new_samples_per_class > 0 and current_strategy != "none":
                    new_train_set_augmented = augment_data(new_train_set, n_new_samples_per_class, [0,1], current_strategy, strategy_params)
                else:
                    new_train_set_augmented = new_train_set
                    
                new_train_set_augmented = Dataset.from_pandas(new_train_set_augmented, split="test")
                print("Training...",end="")

                f1_score, run_time = run_test_job(target=few_shot_model_f1_function, kwargs={"train_set":new_train_set_augmented, "test_set":new_test_set, "model_name":model, "loss":loss, "num_epochs":num_epochs, "batch_size":batch_size, "ratio_frozen_weights":ratio_frozen_weights})
                print("Done")
                results[val].append(f1_score)
                run_times[val].append(run_time)
            except Exception as err:
                print(val, "failed", str(err))
    return results, run_times



#############################################
# Frozen weights ratio test
#############################################

def frozen_ratio_tests(params, train_set, test_set, few_shot_model_f1_function):
    """Run a SetFit frozen weights ratio test

    Args:
        params (dict): Test parameters (number of iterations...)
        train_set (datasets.Dataset): Training set
        test_set (datasets.Dataset): Test set

    Returns:
        results (dict): Test results (F1-scores) for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of F1-scores)
        run_times (dict): Run times for each param variation (e.g. for the N-shots test the keys of the dict might be 3, 5, 10 and for each the value associated to it is an array of the training run times)
    """
    n_shot = params["n_shot"]
    n_iter = params["n_iter"]
    model = params["model"]
    loss = params["loss"]
    input_length_range = params["input_length_range"] if "input_length_range" in params else None
    num_epochs = params["num_epochs"] if "num_epochs" in params else None
    batch_size = params["batch_size"] if "batch_size" in params else None
    ratios_frozen_weights = params["ratio_frozen_weights"] if "ratio_frozen_weights" in params else None
    
    results = {}
    run_times = {}
    for r in ratios_frozen_weights:
        results[r] = []
        run_times[r] = []

    new_test_set = Dataset.from_pandas(test_set, split="test")
 
    progress = 0
    progress_end = n_iter * (len(ratios_frozen_weights))
    start_time = time.time()
    
    for i in range(n_iter):
        if not (input_length_range is None):
            new_train_set = filter_dataset(train_set, input_length_range[0], input_length_range[1])
        else:
            new_train_set = train_set
        new_train_set = Dataset.from_pandas(new_train_set, split="train")
        new_train_set = sample_dataset(new_train_set, label_column="label", num_samples=n_shot, seed=47*i)
        
        for ratio in ratios_frozen_weights:
            progress += 1
            print("Step:", progress, "/", progress_end,"Estimated remaining time:", get_remaining_time_str(start_time, progress, progress_end))
   
            try:
                f1_score, run_time = run_test_job(target=few_shot_model_f1_function, kwargs={"train_set":new_train_set, "test_set":new_test_set, "model_name":model, "loss":loss, "num_epochs":num_epochs, "batch_size":batch_size, "ratio_frozen_weights":ratio})
                results[ratio].append(f1_score)
                run_times[ratio].append(run_time)
            except Exception as err:
                print(ratio, "failed", str(err))
    return results, run_times


#############################################
# Run a task on another process
#############################################

from torch.multiprocessing import Pool, Process, set_start_method
def run_test_job(target, kwargs=None):
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    receiver, sender = multiprocessing.Pipe()

    if not(kwargs is None) and type(kwargs) == type({}):
        args_with_return_val = kwargs
    else:
        args_with_return_val = {}
    args_with_return_val["pipe"] = sender
    
    process = multiprocessing.Process(target=target, kwargs=args_with_return_val)
    process.start()
    result = receiver.recv()
    while(type(result) == type("")): # Redirect stdout while the data received are strings
        print(result, end="")
        result = receiver.recv()
    process.join()
    receiver.close()
    
    if type(result) == Exception:
        raise result
    elif type(result) == type(()):
        return result[0], result[1]
    else:
        raise Exception("Invalid values were returned by the training child process")
    