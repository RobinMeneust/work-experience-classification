## Author: Mathis TEMPO

from flair.data import Sentence, Corpus
from flair.models import TARSClassifier
from flair.trainers import ModelTrainer
import torch
import gc
import os

# Set device to GPU if available, else CPU
if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def get_tars_model():
    """Loads and returns the TARS model, clearing GPU cache first.

    Attempts to load the 'tars-base' model. If unsuccessful, raises an exception.

    Returns:
        TARSClassifier: The loaded TARS model.

    Raises:
        Exception: If the TARS model could not be loaded.
    """
    torch.cuda.empty_cache()
    gc.collect()
    try:
        tars = TARSClassifier.load('tars-base')
    except Exception as e:
        raise Exception("Flair model could not be loaded: ", str(e))
    return tars

def eval(test_set, tars_model):
    """Evaluates TARS model on a test dataset.

    Converts the test set's texts and labels into Sentence objects and evaluates them using the TARS model.

    Args:
        test_set (dict): The test dataset containing 'text' and 'label' pairs.
        tars_model (TARSClassifier): The TARS model to evaluate.

    Returns:
        float: The main score from the evaluation.
    """
    tars_model.eval()
    test_sentences = [Sentence(text).add_label('class', str(label)) for text, label in zip(test_set['text'], test_set['label'])]

    result = tars_model.evaluate(test_sentences, gold_label_type='class', mini_batch_size=32)
    return result.main_score

def flair_train(train_set, tars_model):
    """Trains the TARS model with a training set and saves the trained model.

    Configures the trainer, creates a new classification task, and starts the training process.

    Args:
        train_set (dict): The training dataset containing 'text' and 'label' pairs.
        tars_model (TARSClassifier): The TARS model to train.

    Returns:
        TARSClassifier: The trained TARS model.
    """
    # Create Sentence objects for training and test
    train_sentences = [Sentence(text).add_label('class', str(label)) for text, label in zip(train_set['text'], train_set['label'])]
    
    # Create the corpus
    corpus = Corpus(train=train_sentences)
    unique_labels = list(set(train_set['label']))
    unique_labels = list(map(str, unique_labels))
    
    # Add a new task for labels
    tars_model.add_and_switch_to_new_task(task_name='Classification task', label_dictionary=unique_labels, label_type='class')
    
    # Configure the trainer
    trainer = ModelTrainer(tars_model, corpus)
    
    # Create output folder if it doesn't exist
    output_path = r"../saved_models/flair"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    trainer.train(output_path,
        learning_rate=0.02,
        mini_batch_size=16,
        max_epochs=16,
    )
    
    return tars_model
