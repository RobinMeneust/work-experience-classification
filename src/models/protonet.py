import torch
from torcheval.metrics.functional import binary_f1_score
from transformers import AutoTokenizer, AutoModel
import gc

if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
def get_tokenizer_and_model(model_name="google-bert/bert-base-multilingual-cased", ratio_frozen_weights=0.7):
    """Download and return the tokenizer and the given model used for the embedding

    Args:
        model_name (str, optional): Name of the model to be downloaded. Defaults to "google-bert/bert-base-multilingual-cased".
        ratio_frozen_weights (float, optional): Ratio of weights that will be frozen in the model (used to reduce training memory usage). Defaults to 0.7.

    Returns:
        (any): Tokenizer loaded from the given name
        (any): Model loaded from the given name
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedding_model = AutoModel.from_pretrained(model_name)
    embedding_model.to(device)
    
    # Freeze some parameters if we can't load the whole model in the VRAM
    nb_frozen_params = int(ratio_frozen_weights * len(list(embedding_model.named_parameters())))

    for _, param in list(embedding_model.named_parameters())[0:nb_frozen_params-1]: 
        param.requires_grad = False
    return tokenizer, embedding_model

def gen_tokenized_support_set(n_shots, tokenizer, dataset):
    """Generate the support set but store the tokens instead of the raw text inputs

    Args:
        n_shots (int): Number of examples per class
        tokenizer (any): Tokenizer used to generate the tokens from the texts
        dataset (datasets.Dataset): Dataset containing examples that will be selected and tokenized

    Returns:
        (dict): Dictionnary whose keys are the classes value (e.g. 0, 1) and whose values are the list of tokenized sentences (one item of the list is one tokenized sentence)
    """
    shuffled_dataset = dataset.shuffle(seed=42)
       
    support_set = {}
    for t in [0,1]: # class 0 and class 1 (not related to AI and related to AI)
        current_target_dataset = shuffled_dataset.filter(lambda example, idx: example["label"] == t, with_indices=True)
        support_set[t] = []
        for i in range(n_shots):
            encoded_input = tokenizer(current_target_dataset[i]["text"], return_tensors='pt', truncation=True)
            encoded_input.to(device)
            support_set[t].append(encoded_input)
    return support_set

def get_prototypes_support_set(support_set, embedding_model):
    """Generate the prototypes embeddings of the given support set using the provided embedding model

    Args:
        support_set (dict): Dictionnary whose keys are the classes value (e.g. 0, 1) and whose values are the list of tokenized sentences (one item of the list is one tokenized sentence)
        embedding_model (any): Embedding model used to vectorize the tokens in the support set

    Returns:
        (dict): Dictionnary whose keys are the classes value (e.g. 0, 1) and whose values are the list of embedding vectors
    """
    prototypes_support_set = {}
    for t in support_set.keys():
        embeddings_support_set = []
        for i in range(len(support_set[t])):
            output = embedding_model(**(support_set[t][i]))["pooler_output"]
            embeddings_support_set.append(output)
        prototypes_support_set[t] = torch.mean(torch.stack(embeddings_support_set), axis=0)
    return prototypes_support_set

def predict(tokenizer, embedding_model, instance, support_set):
    """Predict the class of the given instance

    Args:
        tokenizer (any): Tokenizer used to generate the tokens of the text input
        embedding_model (any): Embedding model used to create the input embedding (that will be compared to the prototypes embeddings)
        instance (string): Text input whose class will be determined
        support_set (dict): Dictionnary whose keys are the classes value (e.g. 0, 1) and whose values are the list of tokenized sentences (one item of the list is one tokenized sentence)

    Returns:
        (number): Predicted label
    """
    embedding_model.eval()
    encoded_input = tokenizer(instance, return_tensors='pt', truncation=True)
    encoded_input.to(device)
    embedding = embedding_model(**encoded_input)["pooler_output"]
    del encoded_input
    gc.collect()
    torch.cuda.empty_cache()
    
    similarities = []
    
    prototypes_support_set = get_prototypes_support_set(support_set, embedding_model)
    
    for key in prototypes_support_set.keys():
        similarity_current_key = torch.nn.functional.cosine_similarity(embedding, prototypes_support_set[key])
        similarities.append(similarity_current_key)
    return list(prototypes_support_set.keys())[torch.argmax(torch.stack(similarities))] # Take the closest element of all classes and return its class label


def gen_batches(data, tokenizer, batch_size):
    """Generate batches of tokenized texts

    Args:
        data (datasets.Dataset): Dataset from which batches are generated
        tokenizer (any): Tokenizer used to generate the tokens of the text input
        batch_size (int): Size of the batch to be generated (number of examples)

    Returns:
        (list): List of tuples (tokenized texts, labels of the texts)
    """
    batches = []
    shuffled_set = data.shuffle()

    nb_batches = len(shuffled_set) // batch_size
    if nb_batches == 0:
        nb_batches = 1
        batch_size = len(shuffled_set) % batch_size
    k = 0
    len_shuffled_set = len(shuffled_set)
    unprocessed_data = list(shuffled_set["text"])
    
    for i in range(nb_batches):
        j = 0
        labels = []
        start = i * batch_size
        end = start + batch_size
        unprocessed_batch = unprocessed_data[start:end]
        inputs = tokenizer(unprocessed_batch, return_tensors='pt', padding=True, truncation=True)

        while(j<batch_size and k<len_shuffled_set):
            labels.append(shuffled_set[k]["label"])
            k += 1
            j += 1
        batches.append((inputs, labels))
            
    return batches



def eval(test_set, tokenizer, embedding_model, support_set, verbose=False):
    """Evaluate the ProtoNet model

    Args:
        test_set (datasets.Dataset): Test set
        tokenizer (any): Tokenizer used to generate the tokens of the text input
        embedding_model (any): Embedding model used to create the input embedding (that will be compared to the prototypes embeddings)
        support_set (dict): Dictionnary whose keys are the classes value (e.g. 0, 1) and whose values are the list of tokenized sentences (one item of the list is one tokenized sentence)
        verbose (bool, optional): If true then more messages are displayed (progression). Defaults to False.

    Returns:
        (number): F1-score
    """
    embedding_model.eval()

    predictions = []
    expected = []

    batches = gen_batches(test_set, tokenizer, 16)

    prototypes_support_set = get_prototypes_support_set(support_set, embedding_model)

    progress = 0
    progress_temp = 0
    progress_end = 10
    progress_step = max(len(batches) // progress_end, 1)
    
    for batch in batches:
        inputs, labels = batch
        
        if verbose:
            progress_temp += 1
            
            if progress_temp % progress_step == 0:
                progress += 1
                print("Eval:", progress,"/", progress_end)
        
        inputs.to(device)
        embedding_model_output = embedding_model(**inputs)["pooler_output"]
        del inputs
        gc.collect()
        torch.cuda.empty_cache()
            
        for i in range(len(embedding_model_output)):
            embedding = torch.unsqueeze(embedding_model_output[i],0)
            similarities = []
            for key in prototypes_support_set.keys():
                similarity_current_key = torch.nn.functional.cosine_similarity(embedding, prototypes_support_set[key])
                similarities.append(similarity_current_key)
            predictions.append(torch.tensor(list(prototypes_support_set.keys())[torch.argmax(torch.stack(similarities))])) # Take the closest element of all classes and return its class label
            expected.append(torch.tensor(labels[i]))

    predictions = torch.stack(predictions)
    expected = torch.stack(expected)

    return binary_f1_score(predictions, expected).item()


def protonet_train(support_set, train_set, tokenizer, embedding_model, num_epochs=20, batch_size=16, verbose=False):
    """Train the given model

    Args:
        support_set (dict): Dictionnary whose keys are the classes value (e.g. 0, 1) and whose values are the list of tokenized sentences (one item of the list is one tokenized sentence)
        train_set (datasets.Dataset): _description_
        tokenizer (any): Tokenizer used to generate the tokens of the text input
        embedding_model (any): Embedding model used to create the input embedding (that will be compared to the prototypes embeddings)
        num_epochs (int, optional): _description_. Defaults to 20.
        batch_size (int, optional): _description_. Defaults to 16.
        verbose (bool, optional): If true then more messages are displayed (progression...). Defaults to False.

    Returns:
        (any): Trained model
    """

    optimizer = torch.optim.AdamW(embedding_model.parameters(), lr=1e-5)
    torch.cuda.empty_cache()
    embedding_model.zero_grad()
    
    try:
        embedding_model.train()
        for epoch in range(1,num_epochs+1):
            if verbose:
                print("Epoch: ", epoch, "/", num_epochs,"...",end="")
            batches = gen_batches(train_set, tokenizer, batch_size)
            epoch_mean_loss = 0
            
            for batch in batches:
                optimizer.zero_grad()
                inputs, labels = batch
                inputs.to(device)
                embedding_model_output = embedding_model(**inputs)["pooler_output"]
                
                del inputs
                losses = []           
                
                embeddings_support_set = get_prototypes_support_set(support_set, embedding_model)
            
                for i in range(len(embedding_model_output)):
                    input2 = torch.unsqueeze(embedding_model_output[i],0)
                    input2.to(device)
                    for j in embeddings_support_set.keys():
                        current_class_support_data = embeddings_support_set[j]
                        target = torch.tensor([1.0]) if j == labels[i] else torch.tensor([-1.0])
                        target = target.to(device)
                        losses.append(torch.nn.functional.cosine_embedding_loss(current_class_support_data, input2, target))
                        del target
                    del input2
                
                gc.collect()
                torch.cuda.empty_cache()
                loss = torch.mean(torch.stack(losses))
                epoch_mean_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            if verbose:  
                epoch_mean_loss /= len(batches)
                print(f" Training loss: {epoch_mean_loss:.2f}")
    finally:
        torch.cuda.empty_cache()
    return embedding_model