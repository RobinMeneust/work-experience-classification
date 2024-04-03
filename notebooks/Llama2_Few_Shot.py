import torch
import torch.nn as nn
import replicate
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import pandas as pd
import numpy as np
import random
import time


print("Please select an execution mode")
print("1.Few shot 'classic' ")
print("2.Few shot word length ")
print("3.Few shot language ")
choice = input()
match choice:
		case "1":
			language = 0
			word = 0
		case "2":
			print("Enter minimum word length : ")
			char = input()
			if(int(char) <= 0):
				print("Minimum number of words must be >= 1 ! Exiting . . .")
				exit()
			charMin = int(char)
			print("Enter maximum word length : ")
			char = input()
			if(int(char) < charMin):
				print("Maximum number of words must be >= minimum number of words ! Exiting . . .")
				exit()
			charMax = int(char)
			word = 1
			language = 0
		case "3":
			print("Enter wanted language ('en' or 'fr'): ")
			wantedLanguage = input()
			language = 1
			word = 0
		case _:
			print("Wrong input. Exiting process")
			exit()

print("Enter number of wanted shots : ")
KshotInput = input()
if(int(KshotInput) <= 0):
	print("Number of shots must be >= 1 ! Exiting . . .")
	exit()
Kshot = int(KshotInput)
truePositive = 0
trueNegative = 0
falsePositive = 0
falseNegative = 0

from langdetect import detect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
offload_buffers=True
USE_FLASH_ATTENTION=1

#Load database
dataFrame = pd.read_pickle(r'7587_corrige.pkl')
subset = dataFrame[['jobTitle', 'description', 'label']].copy()

subset.reset_index(drop=True, inplace=True)
subset.replace('', np.nan, inplace=True)
subset.dropna(inplace=True)

subset['text'] = subset['jobTitle'] + '\n' + subset['description']
subset = subset[['text','label']]

#Database getters
def getEntry(i):
	return "\nCLASSIFY:" + subset.iloc[i]["text"]
def getEntryLabel(i):
	return subset.iloc[i]["label"]
def getNumberOfWords(i):
	return len(subset.iloc[i]["text"].split())
#Load model
model = AutoModelForCausalLM.from_pretrained(
	"meta-llama/Llama-2-7b-chat-hf",
	cache_dir = "./Llama-2-7b-chat-hf",
	device_map = 'auto',
	token = "XXX" #Insert hugging face token here
)

#Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
	"meta-llama/Llama-2-7b-chat-hf",
	cache_dir="./Llama-2-7b-chat-hf",
	token = "XXX" #Insert hugging face token here
)

#Get llama 2 limited answer
def getAnswer(prompt, maxTokens):
	inputs = tokenizer(prompt, return_tensors="pt").to(device)
	outputs = model.generate(**inputs, max_new_tokens=maxTokens)

	answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
	return answer

#Llama 2 text generation
def ChatCompletion(prompt, system_prompt=None):
	output = replicate.run(
	model,
	input={"system_prompt": system_prompt,
			"prompt": prompt,
			"max_new_tokens":1000}
	)
	return "".join(output)

#F1-score calculator
def f1Score(tp, fp, fn):
	return (2*tp) / (2*(tp+fp+fn))

#Initialisation is done. Starting tests and timer
start_time = time.time()

for i in range(10):
	prompt = ""
	numberOfYes = 0
	numberOfNo = 0
	KshotLoop = Kshot*2
	i = 0
	j = 0
	
	#Generation of the "classic" support set
	if(language == 0 and word == 0):
		while i < KshotLoop:
			entry = random.randrange(5166)
			#Check if the entry class (ai or not) and if we have enough
			if getEntryLabel(entry) < 3.0 and numberOfNo != Kshot:
				prompt += getEntry(entry)
				numberOfNo += 1
				prompt += "\nANSWER: Non"
			elif getEntryLabel(entry) >= 3.0 and numberOfYes != Kshot:
				prompt += getEntry(entry)
				numberOfYes += 1
				prompt += "\nANSWER: Oui"

			#In case one of the two classes is full of examples we continue the search
			if numberOfNo == Kshot or numberOfYes == Kshot:
				if numberOfNo == Kshot and numberOfYes == Kshot: #Exit condition
					break
				KshotLoop += 1 #else we continue
			i += 1
	#Generation of the specific language support set
	elif(language == 1):
		while i < KshotLoop:
			entry = random.randrange(5166)
			#Check if the entry is of the selected language
			if detect(subset.iloc[entry]["text"]) == wantedLanguage:
				#Check if the entry class (ai or not) and if we have enough
				if getEntryLabel(entry) < 3.0 and numberOfNo != Kshot:
					prompt += getEntry(entry)
					numberOfNo += 1
					prompt += "\nANSWER: Non"
				elif getEntryLabel(entry) >= 3.0 and numberOfYes != Kshot:
					prompt += getEntry(entry)
					numberOfYes += 1
					prompt += "\nANSWER: Oui"

				#In case one of the two classes is full of examples we continue the search
				if numberOfNo == Kshot or numberOfYes == Kshot:
					if numberOfNo == Kshot and numberOfYes == Kshot: #Exit condition
						break
					KshotLoop += 1 #else we continue
				i += 1
	#Generation of the word length support set
	else:
		while i < KshotLoop:
			entry = random.randrange(5166)
			#Check if the entry is of an adequate length
			if getNumberOfWords(entry) <= charMax and getNumberOfWords(entry) >= charMin:
				#Check if the entry class (ai or not) and if we have enough
				if getEntryLabel(entry) < 3.0 and numberOfNo != Kshot:
					prompt += getEntry(entry)
					numberOfNo += 1
					prompt += "\nANSWER: Non"
				elif getEntryLabel(entry) >= 3.0 and numberOfYes != Kshot:
					prompt += getEntry(entry)
					numberOfYes += 1
					prompt += "\nANSWER: Oui"

				#In case one of the two classes is full of examples we continue the search
				if numberOfNo == Kshot or numberOfYes == Kshot:
					if numberOfNo == Kshot and numberOfYes == Kshot: #Exit condition
						break
					KshotLoop += 1 #else we continue
				i += 1

	#Generation of the querry set
	if(language == 1):

		#Génération of the specific language querry set
		loop2 = 5
		while j < loop2:
			entry = random.randrange(5166)
			if detect(subset.iloc[entry]["text"]) == wantedLanguage:
				prompt += getEntry(entry)
				prompt += "\nANSWER:"
				break
			else:
				loop2 += 1
			j += 1
	else:
		#Génération of the querry set
		entry = random.randrange(5166)
		prompt += getEntry(entry)
		prompt += "\nANSWER:"


	#Getting Llama 2 answer
	answer = getAnswer(prompt, 2)
	print(answer)

	verif = getEntryLabel(entry)
	print("\nQuery set label (debug): " + str(verif))


	print("Llama 2 answer (debug): *"+answer[-3:]+"*")

	#Saving llama 2 answers quality
	if (answer[-3:] == "Oui" or answer[-3:] =="es\n" or answer[-3:] == "Yes") and verif >= 3.0:
		truePositive += 1
		print("True positive")
	elif (answer[-3:] == "Oui" or answer[-3:] =="es\n" or answer[-3:] == "Yes") and verif < 3.0:
		print("False positive")
		falsePositive += 1

	if (answer[-3:] == "Non " or answer[-3:] == "on\n" or answer[-3:] == "Non" or answer[-3:] == " No") and verif < 3.0:
		trueNegative += 1
		print("True negative")
	elif (answer[-3:] == "Non " or answer[-3:] == "on\n" or answer[-3:] == "Non" or answer[-3:] == " No") and verif >= 3.0:
		print("False negative")
		falseNegative += 1

#Process if done. Print results and timer
print("--- %s seconds ---" % (time.time() - start_time))
print("True positive  : " + str(truePositive))
print("False positive : " + str(falsePositive))
print("True negative  : " + str(trueNegative))
print("False negative : " + str(falseNegative))
print("F1-Score : " + str(f1Score(truePositive, falsePositive, trueNegative)))