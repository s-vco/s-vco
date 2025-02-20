import argparse
import json
import re

parser = argparse.ArgumentParser(description='Process JSONL file path.')
parser.add_argument('--answer_file', default = "answer.jsonl",help='Path to the JSONL file')
args = parser.parse_args()

def is_answer_correct(correct_label, llm_answer):
    # Remove extraneous characters like parentheses and whitespace from both correct label and LLM's answer
    llm_answer_cleaned = re.sub(r'[\(\)\s]', '', llm_answer.strip().lower()[:4])
    correct_label_cleaned = re.sub(r'[\(\)\s]', '', correct_label.strip().lower())
    # Check if the cleaned LLM answer matches the cleaned correct label
    return llm_answer_cleaned == correct_label_cleaned

source_to_correct = {}
task_to_correct = {}
source_to_total = {}
task_to_total = {}

with open(args.answer_file, 'r') as file:
    for line in file:
        data = json.loads(line)
        source, task, correct_answer, model_response = data["source"], data["task"], data["answer"], data["response"]
        correct = is_answer_correct(correct_answer, model_response)
        if correct:
            source_to_correct[source] = source_to_correct.get(source, 0) + 1
            task_to_correct[task] = task_to_correct.get(task, 0) + 1
        source_to_total[source] = source_to_total.get(source, 0) + 1
        task_to_total[task] = task_to_total.get(task, 0) + 1

# source acc.
accuracy_2d_ade = source_to_correct['ADE20K'] / source_to_total['ADE20K']
accuracy_2d_coco = source_to_correct['COCO'] / source_to_total['COCO']
accuracy_3d_omni = source_to_correct['Omni3D'] / source_to_total['Omni3D']
# Calculate the accuracy for each type
accuracy_2d = (accuracy_2d_ade + accuracy_2d_coco) / 2
accuracy_3d = accuracy_3d_omni
# Compute the combined accuracy as specified
combined_accuracy = (accuracy_2d + accuracy_3d) / 2

# Print the results
print(f"Combined Accuracy: {combined_accuracy:.3f}")
print()
print(f"Type Accuracies:")
print(f"2D Accuracy: {accuracy_2d:.3f}")
print(f"3D Accuracy: {accuracy_3d:.3f}")
print()
print(f"Source Accuracies:")
print(f"ADE20K Accuracy: {accuracy_2d_ade:.3f}")
print(f"COCO Accuracy: {accuracy_2d_coco:.3f}")
print(f"Omni3D Accuracy: {accuracy_3d_omni:.3f}")
print()
print(f"Task Accuracies:")
for k in task_to_total.keys():
    acc = task_to_correct[k] / task_to_total[k]
    print(f"{k} Accuracy: {acc:.3f}")

