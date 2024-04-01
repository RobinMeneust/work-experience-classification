## Author: Mathis TEMPO

from flair.data import Sentence, Corpus
from flair.models import TARSClassifier
from flair.trainers import ModelTrainer
from torch.utils.data import DataLoader as TorchDataLoader
import torch
import gc
import os

if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
def get_tars_model():
    torch.cuda.empty_cache()
    gc.collect()
    tars = None
    try:
        tars = TARSClassifier.load('AyoubChLin/ESG-bert-BBC_news')
        print("success")
    except Exception as e:
        raise Exception("Flair model could not be loaded: ", str(e))
    return tars

def eval(test_set, tars_model, verbose=False):   
    tars_model.eval()
 
    test_sentences = [Sentence(text).add_label('class', str(label)) for text, label in zip(test_set['text'], test_set['label'])]
    test_loader = TorchDataLoader(test_sentences, batch_size=16, collate_fn=lambda x: x)

    result, _ = tars_model.evaluate(test_loader, gold_label_type='class')
    print(f"F1 Score: {result.main_score}")

    return result.main_score


def flair_train(train_set, tars_model, verbose=False):
    # Create Sentence objects for training and test
    train_sentences = [Sentence(text).add_label('class', str(label)) for text, label in zip(train_set['text'], train_set['label'])]
    
    # Create the corpus
    corpus = Corpus(train=train_sentences)
    unique_labels = train_set['label'].unique().astype(str).tolist()

    # Add a new task for labels
    tars_model.add_and_switch_to_new_task(task_name='Classification task', label_dictionary=unique_labels, label_type='class')

    # Configure the trainer
    trainer = ModelTrainer(tars_model, corpus)

    # Create output folder if it doesn't exist
    output_path = r"../models/flair"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    trainer.train(output_path,
        learning_rate=0.02,
        mini_batch_size=8,
        max_epochs=8,
    )
    
    return tars_model