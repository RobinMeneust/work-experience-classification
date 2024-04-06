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

 # Clear GPU cache, load and return the TARS model
def get_tars_model():
    torch.cuda.empty_cache()
    gc.collect()
    tars = None
    try:
        tars = TARSClassifier.load('tars-base')
    except Exception as e:
        raise Exception("Flair model could not be loaded: ", str(e))
    return tars

 # Evaluate TARS model on test set, return main score
def eval(test_set, tars_model, verbose=False):
    tars_model.eval()
    test_sentences = [Sentence(text).add_label('class', str(label)) for text, label in zip(test_set['text'], test_set['label'])]

    result = tars_model.evaluate(test_sentences, gold_label_type='class', mini_batch_size=32)
    return result.main_score

 # Train TARS model with train set, configure trainer, and save model
def flair_train(train_set, tars_model, verbose=False):
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
        mini_batch_size=8,
        max_epochs=8,
    )
    
    return tars_model
