# main.py
from data_loader import FUNSDDataset
from model_setup import get_model
from train_model import train_model
from transformers import LayoutLMTokenizer

def main():
    tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
    labels_dict = {'header': 0, 'question': 1, 'answer': 2, 'other': 3}
    data_dir = '/home/os/kaggle-dataset/layoutlm/dataset/training_data'  # Correct path

    train_dataset = FUNSDDataset(tokenizer, labels_dict, data_dir)
    num_labels = len(labels_dict)
    model = get_model(num_labels)
    train_model(train_dataset, model,tokenizer)

if __name__ == '__main__':
    main()


