import torch

import sys
from sentence_transformers.losses import BatchHardTripletLossDistanceFunction
from transformers import PrinterCallback, ProgressCallback
from setfit import Trainer, TrainingArguments, SetFitModel
import gc
import time

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
    run_time = None
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
            run_time = time.time() - start_time
            
            metrics = trainer.evaluate()
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
        pipe.send((metrics['f1'], run_time))
    pipe.close()


#############################################
# ProtoNet init and training
#############################################

import protoNet

# Run a test on protonet (training + evaluation)
def protonet_f1_score(train_set, test_set, pipe, model_name, loss=None, distance_metric = None, num_epochs = None, batch_size = None, head_learning_rate = None, ratio_frozen_weights=None):
    sys.stdout = PipeWriter(pipe)
    
    f1_score = None
    run_time = None
    tokenizer = None
    model = None
    err = None
    
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
            
            tokenizer, model = protoNet.get_tokenizer_and_model(model_name, ratio_frozen_weights)
            
            support_set = protoNet.gen_support_set(len(train_set)//2, tokenizer, train_set) # TODO // 2 MUST BE CHANGED to // nbClasses
            
            start_time = time.time()
            model = protoNet.protonet_train(support_set, train_set, tokenizer, model, num_epochs=num_epochs[0], batch_size=batch_size[0])
            run_time = time.time() - start_time
            f1_score = protoNet.eval(test_set, tokenizer, model, support_set)
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
        pipe.send((f1_score, run_time))
    pipe.close()