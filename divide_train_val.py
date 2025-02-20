import json
import argparse
import random

def split_dataset(input_file, val_num, train_file, val_file):
    with open(input_file, "r") as f:
        data = json.load(f)
    random.shuffle(data)
    val_data = data[:val_num]
    train_data = data[val_num:]
    with open(train_file, "w") as f:
        json.dump(train_data, f, indent=4)
    with open(val_file, "w") as f:
        json.dump(val_data, f, indent=4)
    print(f"Dataset split completed.\nTraining set: {len(train_data)} samples\nValidation set: {len(val_data)} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into training and validation sets.")
    parser.add_argument("--input_file", type=str, default="data/mvc.json", help="Path to the input dataset file.")
    parser.add_argument("--val_num", type=int, default=240, help="Number of samples for validation.")
    parser.add_argument("--train_file", type=str, default="data/mvc_train.json", help="Output path for the training set.")
    parser.add_argument("--val_file", type=str, default="data/mvc_val.json", help="Output path for the validation set.")
    
    args = parser.parse_args()
    split_dataset(args.input_file, args.val_num, args.train_file, args.val_file)