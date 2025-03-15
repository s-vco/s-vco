import argparse
import json
import openai
import re
import time
# Create the parser
parser = argparse.ArgumentParser(description='Process JSONL file path.')

# Add arguments
parser.add_argument('--answer_file', default = "answer.jsonl",help='Path to the JSONL file')

# Parse arguments
args = parser.parse_args()

def is_answer_correct(correct_label, llm_answer):
    # Remove extraneous characters like parentheses and whitespace from both correct label and LLM's answer
    llm_answer_cleaned = re.sub(r'[\(\)\s]', '', llm_answer.strip().lower()[:4])
    correct_label_cleaned = re.sub(r'[\(\)\s]', '', correct_label.strip().lower())
    # Check if the cleaned LLM answer matches the cleaned correct label
    return llm_answer_cleaned == correct_label_cleaned

num_correct, num_total = 0, 0
# Continue with the processing of the JSONL file
with open(args.answer_file, 'r') as file:
    index = 0
    for line in file:
        data = json.loads(line)
        question, correct_answer, model_response = data["prompt"], data["answer"], data["response"]
        correct = is_answer_correct(correct_answer, model_response)
        if index %2 ==0:
            last_correct = correct
        if index %2 ==1:
            if last_correct and correct:
                num_correct += 1
                print(f"Paird Acc on {index} & {index-1}: Correct")
            else:
                print(f"Paird Acc on {index} & {index-1}: Wrong")
            num_total += 1
        index += 1
assert num_total == 150
print(f"The paired accuracy is {round(num_correct/num_total, 3)}")