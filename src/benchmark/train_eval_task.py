import torch
import sys
from sentence_transformers.losses import BatchHardTripletLossDistanceFunction
from transformers import PrinterCallback, ProgressCallback
from setfit import Trainer, TrainingArguments, SetFitModel
import gc
import time
import models.flair as flair
import models.protonet as protonet
import models.llama2 as llama2
import multiprocessing

#############################################
# SetFit init and training
#############################################

# Get a setfit model from name or path
def get_setfit_model(model_name, device, use_differentiable_head=False):
    """Get the transformer model from the HuggingFace hub from its name (and download it if it's not on the current machine)

    Args:
        model_name (string): Name of the transformer to be fetched
        use_differentiable_head (bool, optional): False if we want to use the default Logistic classification head and false if we want to use the PyTorch's one (needed for multiclass classification). Defaults to False.

    Returns:
        SetFitModel : SetFit model created with the given transformer
    """
    model = SetFitModel.from_pretrained(model_name, use_differentiable_head=use_differentiable_head)
    gc.collect()
    torch.cuda.empty_cache()
    
    return model.to(device)

# Init setfit trainer
def init_setfit_trainer(model, loss, train_dataset, test_dataset, distance_metric = BatchHardTripletLossDistanceFunction.cosine_distance, num_epochs = (1,16), batch_size = (16,2), head_learning_rate = 1e-2):
    """Initialize the trainer with the test params

    Args:
        model (SetFitModel): SetFit model that will be train with this new trainer
        loss (object): Loss function used (e.g. CosineSimilarityLoss)
        train_dataset (datasets.Dataset): Training set
        test_dataset (datasets.Dataset): Test set
        distance_metric (object, optional): Distance used to compare a pair/triplet of embeddings. Defaults to BatchHardTripletLossDistanceFunction.cosine_distance.
        num_epochs (tuple, optional): Number of epochs: (body_num_epochs, head_num_epochs). Defaults to (1,16).
        batch_size (tuple, optional): Size of the batches: (body_batch_size, head_batch_size). Defaults to (16,2).
        head_learning_rate (number, optional): Head learning rate. Defaults to 1e-2.
    """
    
    trainer_arguments = TrainingArguments(
        show_progress_bar=False,
        loss=loss,
        distance_metric=distance_metric,
        batch_size=batch_size,
        num_epochs=num_epochs,
		head_learning_rate=head_learning_rate,
    )

    trainer = Trainer(
        model=model,
        args=trainer_arguments,
        metric='f1',
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    
    # Disable some logs because there were too many messages during the tests
    trainer.pop_callback(PrinterCallback)
    trainer.pop_callback(ProgressCallback)
    
    return trainer

# We need a class with a write function like sys.stdout to redirect the stdout of the child to the parent
class PipeWriter:
    def __init__(self, pipe):
        self.pipe = pipe

    def write(self, message):
        self.pipe.send(message)
        
    def flush(self):
        pass

    
# Run a test on setfit (training + evaluation)
def setfit_f1_score(train_set, test_set, model_name, loss, pipe, distance_metric = None, num_epochs = None , batch_size = None, head_learning_rate = None, ratio_frozen_weights=None):
    """Initialize and test a SetFit model with the given params

    Args:
        train_dataset (datasets.Dataset): Training set
        test_dataset (datasets.Dataset): Test set
        model_name (string): Name of the transformer to be fetched
        loss (object): Loss function used (e.g. CosineSimilarityLoss)
        distance_metric (object, optional): Distance used to compare a pair/triplet of embeddings. Defaults to None.
        num_epochs (tuple, optional): Number of epochs: (body_num_epochs, head_num_epochs). Defaults to None.
        batch_size (tuple, optional): Size of the batches: (body_batch_size, head_batch_size). Defaults to None.
        head_learning_rate (number, optional): Head learning rate. Defaults to None.

    Raises:
        Exception: If the training set or test set contain only one example, since at least are required by SetFit to create sentence pairs

    Returns:
        number: F1-score
        number: Run time (training only)
    """
    
    sys.stdout = PipeWriter(pipe)
    
    if torch.cuda.is_available():    
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    err = None
    model = None
    trainer = None
    metrics = None
    train_time = None
    eval_time = None
    
    try:
        try:
            if len(train_set) <= 1 or len(test_set) <= 1:
                raise Exception("Invalid data sets length")
            
            # Replace by None with default values
            if distance_metric is None:
                distance_metric = BatchHardTripletLossDistanceFunction.cosine_distance
            if num_epochs is None:
                num_epochs = (1,16)
            if batch_size is None:
                batch_size = (16,2)
            if head_learning_rate is None:
                head_learning_rate = 1e-2
                
            model = get_setfit_model(model_name, device, (not (num_epochs is None)) and num_epochs[1]>1)
            trainer = init_setfit_trainer(model, loss, train_set, test_set, distance_metric, num_epochs, batch_size, head_learning_rate)
            
            start_time = time.time()
            trainer.train()
            train_time = time.time() - start_time
            
            start_time = time.time()
            metrics = trainer.evaluate()
            eval_time = time.time() - start_time
        except Exception as e:
            raise e
        finally:
            del model
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as e2:
        err = e2
    if not(err is None):
        pipe.send(Exception(str(err)))
    else:
        pipe.send((metrics['f1'], train_time, eval_time))
    pipe.close()


#############################################
# ProtoNet init and training
#############################################

# Run a test on protonet (training + evaluation)
def protonet_f1_score(train_set, test_set, pipe, model_name, loss=None, distance_metric = None, num_epochs = None, batch_size = None, head_learning_rate = None, ratio_frozen_weights=None):
    sys.stdout = PipeWriter(pipe)
    
    f1_score = None
    tokenizer = None
    model = None
    err = None
    train_time = None
    eval_time = None
    
    try:
        try:
            if len(train_set) <= 1 or len(test_set) <= 1:
                raise Exception("Invalid data sets length")
            
            # Replace by None with default values
            if num_epochs is None:
                num_epochs = (20,0)
            if batch_size is None:
                batch_size = (4,0)
            if ratio_frozen_weights is None:
                ratio_frozen_weights = 0.7
            
            tokenizer, model = protonet.get_tokenizer_and_model(model_name, ratio_frozen_weights)
            
            support_set = protonet.gen_tokenized_support_set(len(train_set)//2, tokenizer, train_set) # TODO // 2 MUST BE CHANGED to // nbClasses if we want to work on multi class classification
            
            start_time = time.time()
            model = protonet.protonet_train(support_set, train_set, tokenizer, model, num_epochs=num_epochs[0], batch_size=batch_size[0])
            train_time = time.time() - start_time
            
            start_time = time.time()
            f1_score = protonet.eval(test_set, tokenizer, model, support_set)
            eval_time = time.time() - start_time
        except Exception as e:
            raise e
        finally:
            del tokenizer
            del model
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as e2:
        err = e2
        
    if not(err is None):
        pipe.send(Exception(str(err)))
    else:
        pipe.send((f1_score, train_time, eval_time))
    pipe.close()
    
    

#############################################
# Flair init and training
#############################################

# Run a test on Flair (training + evaluation)
def flair_f1_score(train_set, test_set, pipe, model_name=None, loss=None, distance_metric = None, num_epochs = None, batch_size = None, head_learning_rate = None, ratio_frozen_weights=None):
    sys.stdout = PipeWriter(pipe)
    
    f1_score = None
    err = None
    model = None
    train_time = None
    eval_time = None
    
    try:
        try:
            if len(train_set) <= 1 or len(test_set) <= 1:
                raise Exception("Invalid data sets length")
            
            model = flair.get_tars_model()
            
            start_time = time.time()
            flair.flair_train(train_set, model, verbose=False)
            train_time = time.time() - start_time

            start_time = time.time()
            f1_score = flair.eval(test_set, model)
            eval_time = time.time() - start_time
        except Exception as e:
            raise e
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as e2:
        err = e2
        
    if not(err is None):
        pipe.send(Exception(str(err)))
    else:
        pipe.send((f1_score, train_time, eval_time))
    pipe.close()
    

#############################################
# Llama2 init and training
#############################################

# Run a test on Llama2 (training + evaluation)
def llama2_f1_score(train_set, test_set, pipe, model_name=None, loss=None, distance_metric = None, num_epochs = None, batch_size = None, head_learning_rate = None, ratio_frozen_weights=None):
    sys.stdout = PipeWriter(pipe)
    
    f1_score = None
    tokenizer = None
    model = None
    err = None
    eval_time = None
    
    try:
        try:
            if len(train_set) <= 1 or len(test_set) <= 1:
                raise Exception("Invalid data sets length")
            
            tokenizer, model = llama2.get_tokenizer_and_model()
            
            support_set = llama2.gen_support_set(len(train_set)//2, tokenizer, train_set) # TODO // 2 MUST BE CHANGED to // nbClasses if we want to work on multi class classification
            
            start_time = time.time()
            f1_score = llama2.eval(test_set, tokenizer, model, support_set, True)
            eval_time = time.time() - start_time
        except Exception as e:
            raise e
        finally:
            del tokenizer
            del model
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as e2:
        err = e2
        
    if not(err is None):
        pipe.send(Exception(str(err)))
    else:
        pipe.send((f1_score, 0, eval_time))
    pipe.close()
    
    

#############################################
# Run a task on another process
#############################################

from torch.multiprocessing import set_start_method
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
        return result
    else:
        raise Exception("Invalid values were returned by the training child process")