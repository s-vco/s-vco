import argparse
import json
import re

parser = argparse.ArgumentParser(description='Process JSONL file path.')
parser.add_argument('--answer_file', default = "answer.jsonl",help='Path to the JSONL file')
args = parser.parse_args()

def is_answer_correct(question, correct_label, llm_answer):
    # RealworldQA --> NOT ALL MCQA-format!!!
    # Remove extraneous characters like parentheses, whitespace, and periods from both correct label and LLM's answer
    if "answer directly with only the letter of the correct option and nothing else" in question.lower():
        llm_answer_cleaned = re.sub(r'[\(\)\s\.]', '', llm_answer.strip().lower()[:4])
    else:
        llm_answer_cleaned = re.sub(r'[\(\)\s\.]', '', llm_answer.strip().lower())
    correct_label_cleaned = re.sub(r'[\(\)\s\.]', '', correct_label.strip().lower())
    # Check if the cleaned LLM answer matches the cleaned correct label
    return llm_answer_cleaned == correct_label_cleaned

num_correct, num_total = 0, 0

with open(args.answer_file, 'r') as file:
    index = 0
    for line in file:
        data = json.loads(line)
        question, correct_answer, model_response = data["prompt"], data["answer"], data["response"]
        correct = is_answer_correct(question, correct_answer, model_response)
        index += 1
        if correct:
            num_correct += 1
        num_total += 1

print(f"Num of correct is {num_correct}")
print(f"Num of total is {num_total}")
print(f"The accuracy is {round(num_correct/num_total, 3)}")
