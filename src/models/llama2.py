import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def get_tokenizer_and_model():
    #Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        cache_dir="../saved_models/Llama-2-7b-chat-hf",
        token=True
    )
    
    #Load model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        cache_dir = "../saved_models/Llama-2-7b-chat-hf",
        device_map = 'auto',
        token=True
    )
    #model.to(device)
    return tokenizer, model

offload_buffers=True
USE_FLASH_ATTENTION=1


#Get llama 2 limited answer
def getAnswer(prompt, maxTokens, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt")#.to(device)
    outputs = model.generate(**inputs, max_new_tokens=maxTokens)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

#F1-score calculator
def f1Score(tp, fp, fn):
    return (2*tp) / (2*(tp+fp+fn))

def gen_support_set(n_shots, dataset):
    shuffled_dataset = dataset.shuffle(seed=42)
    support_set = {}
    
    for t in [0,1]: # class 0 and class 1 (not related to AI and related to AI)
        current_target_dataset = shuffled_dataset.filter(lambda example, idx: example["label"] == t, with_indices=True)
        support_set[t] = []
        for i in range(n_shots):
            support_set[t].append(current_target_dataset[i]["text"])
    return support_set

def eval(test_set, tokenizer, model, support_set, verbose=False):
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    
    support_set_prompt = ""
    
    for label in support_set.keys():
        for text in support_set[label]:
            support_set_prompt += "\nCLASSIFY:" + text
            support_set_prompt += "\nANSWER: " + ("Oui" if label == 1 else "Non")
        
    
    progress = 0
    progress_temp = 0
    progress_end = min(10,len(test_set))
    progress_step = max(len(test_set) // progress_end, 1)
    
    for i in range(len(test_set)):
        if verbose:
            progress_temp += 1
            
            if progress_temp % progress_step == 0:
                progress += 1
                print("Eval:", progress,"/", progress_end)
        
        prompt = support_set_prompt
        
        #Generation of the querry set
        prompt += "\nCLASSIFY:" + test_set[i]["text"] + "\nANSWER:"

        #Getting Llama 2 answer
        answer = getAnswer(prompt, 2, tokenizer, model)
        # print(answer)

        expected = int(test_set[i]["label"])

        #Saving llama 2 answers quality
        if (answer[-3:] == "Oui" or answer[-3:] =="es\n" or answer[-3:] == "Yes") and expected == 1:
            truePositive += 1
        elif (answer[-3:] == "Oui" or answer[-3:] =="es\n" or answer[-3:] == "Yes") and expected == 0:
            falsePositive += 1

        if (answer[-3:] == "Non " or answer[-3:] == "on\n" or answer[-3:] == "Non" or answer[-3:] == " No") and expected == 0:
            trueNegative += 1
        elif (answer[-3:] == "Non " or answer[-3:] == "on\n" or answer[-3:] == "Non" or answer[-3:] == " No") and expected == 1:
            falseNegative += 1
        
    return f1Score(truePositive, falsePositive, trueNegative)