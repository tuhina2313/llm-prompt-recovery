# Script to generate target texts from given source text and rewrite prompt using Llama2 from the Together.ai API
import os
import json
from together import Together
import nltk
import requests
import csv
import random
from nltk.tokenize import sent_tokenize
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")

def create_file(output_file, input, rewrite, label, dataset, target_texts):
	print("Create CSV function called")

	# Specify the CSV file path
	csv_file_path = output_file

	# Write extracted fields to CSV file
	with open(csv_file_path, 'w', newline='') as csvfile:
		fieldnames = ['OriginalText', 'RewritePrompt', 'TargetText', 'RewriteLabel', 'DatasetLabel']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

		# Write header
		writer.writeheader()

		# Write data
		for i, entry in enumerate(input):
			# entry_row = json.loads(entry)
			writer.writerow({
				'OriginalText': entry,
				'RewritePrompt': rewrite[i],
				'TargetText': target_texts[i],
				'RewriteLabel': label[i],
				'DatasetLabel': dataset[i]
			})

	print(f'CSV file "{csv_file_path}" has been created.')

def toxicity_classifier(text):
    pipe = pipeline("text-classification", model="unitary/toxic-bert")
    sequences = pipe(text)
    print(sequences)

def extract_target_text(input_text):
    target_index = input_text.find('Target Text:')
    if target_index == -1:
        return None  # 'Target Text:' not found in input_text
    
    target_text = input_text[target_index + len('Target Text:'):]
    
    target_text = "Target Text: " + target_text.strip()
    
    return target_text

#use Together.ai to generate 5k data samples for training
def call_together_api(prompts):
    all_responses = [] #store outputs
    TOGETHER_API_KEY = " "
    client = Together(api_key=TOGETHER_API_KEY)
    count = 0 #progress track
    for prompt in prompts:
        count = count+1
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-8b-chat-hf",
            messages=[{"role": "user", "content": prompt}],
        )
        all_responses.append(response.choices[0].message.content)
        # all_responses.append(clean_text)
        print(count)
        # print("_________________________________________")
    return all_responses

# function to format the prompts for data generation. The format is as follows: 
# <system instructions>
# Rewrite Prompt: <randomly sampled rewrite prompt>
# Original text: <the input text on which modification is required>
def create_llama_prompts(original_texts, rewrite_file, dataset_labels):
    rewrite_prompts, input, rewrite, label, dataset = [], [], [], [], []
    all_input_prompts = []
    system_prompt = "You are an AI assistant helping a user convert a given source text using a rewrite prompt. Make sure the target text is similar in length to source text and do not add any new information. Generate only the target text, do not output any emojis in the target text and use the following format for output: Target Text: <Target Text>.\n"
    #some examples of the task are as follows: \n Source Text: Mars, the fourth planet from the Sun, has captivated humanity's imagination for centuries. Its rusty-red hue and enigmatic landscape have inspired countless exploratory missions. \n Rewrite Prompt: Rewrite this text as a horror movie plot. \n Target Text: Mars, the blood-red planet, holds dark secrets that have plagued humanity for centuries. Its desolate landscape lures explorers to their doom, shrouding their fate in mystery and horror.

    with open(rewrite_file, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)

        for prompt in csv_reader:
            rewrite_prompts.append((str(prompt[0]),prompt[1]))
            a, b = prompt[0], prompt[1]

        for i, original_text in enumerate(original_texts):
            if len(original_text) > 20:
                rewrite_prompt = random.choice(rewrite_prompts)
                inp_prompt = system_prompt + "\nRewrite prompt: " + rewrite_prompt[0] + "\nSource Text: " + original_text
                all_input_prompts.append(inp_prompt)
                input.append(original_text)
                rewrite.append(rewrite_prompt[0])
                label.append(rewrite_prompt[1])
                dataset.append(dataset_labels[i])
    return all_input_prompts, input, rewrite, label, dataset



def truncate_paragraph(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    # sample randomly for diverse lengths 
    length = [3, 4, 5, 6, 7]
    max_sentences = random.choice(length)
    # Truncate the sentences to the specified maximum
    truncated_text = ' '.join(sentences[:max_sentences])

    return truncated_text

def load_data(filename):
    original_texts = []
    dataset_labels = []
    f = open(filename)
    
    data = json.load(f)
    # Specifically for Reddit ELI5 data, there are some human responses that are toxic and we do not want those in our dataset
    # An off-the-shelf toxicity classifier is used to weed out toxic responses 
    # For cases where none of the human responses are usable, we use chatgpt-answers from the data, clip them and use them as source text
    pipe = pipeline("text-classification", model="unitary/toxic-bert")

    for row in data:
        for answer in row['human_answers']:
            text_to_append = str(answer)
            res_len = len(sent_tokenize(text_to_append))
            if res_len > 6:
                text_to_append = ' '.join(answer[:4])
            toxicity_score = pipe(text_to_append)
            if toxicity_score[0]['score'] < 0.5:
                original_texts.append(text_to_append)
                dataset_labels.append("eli5-human")
                break
            else:
                text_to_append = row['chatgpt_answers']
                text_to_append = ' '.join(text_to_append[:4])
                original_texts.append(text_to_append)
                dataset_labels.append("eli5-chatgpt")
    for row in data:
        res_len = len(sent_tokenize(text_to_append))
        if res_len > 6:
            text_to_append = ' '.join(answer[:4])
        text_to_append = str(row['chatgpt_answers'])
        text_to_append = truncate_paragraph(text_to_append)
        original_texts.append(text_to_append)
        dataset_labels.append("eli5-chatgpt")

    f.close()
    return original_texts, dataset_labels

if __name__=="__main__":
    original_texts, dataset_labels = load_data('/Users/tuhinatripathi/Desktop/UMass/CognitiveLoad/unity-files/eli5_last750.json')
    rewrite_file = '/Users/tuhinatripathi/Desktop/UMass/Spring24/NLP/Project/all_rewrite.csv'
    all_input_prompts, input, rewrite, label, dataset = create_llama_prompts(original_texts, rewrite_file, dataset_labels)
    print((all_input_prompts[0]))
    target_texts = call_together_api(all_input_prompts)
    # target_gpt = call_gpt_api(all_input_prompts)

    output_csv = 'output_eli5_last750.csv'
    create_file(output_csv, input, rewrite, label, dataset, target_texts)
    print("Finished")


    