import json
import os
import torch
from transformers import LayoutLMTokenizer
from PIL import Image

class FUNSDDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, labels, data_dir, max_length=512):
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_length = max_length
        self.data = []
        self.image_path = os.path.join(data_dir, 'images')  # Assuming images are stored here
        annotations_dir = os.path.join(data_dir, 'annotations')
        for file_name in os.listdir(annotations_dir):
            if file_name.endswith('.json'):
                file_path = os.path.join(annotations_dir, file_name)
                image_file = os.path.join(self.image_path, file_name.replace('.json', '.png'))
                with open(file_path, 'r') as file:
                    try:
                        data = json.load(file)
                        self.data.append((data, image_file))
                        print(f"Successfully loaded {file_name}")
                    except json.JSONDecodeError as e:
                        print(f"Failed to decode JSON from {file_name}: {e}")
        
        print(f"Total files loaded: {len(self.data)}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item, image_file = self.data[idx]
        image = Image.open(image_file).convert("RGB")
        width, height = image.size

        try:
            texts, boxes, actual_labels = self.extract_data(item)
            normalized_boxes = [self.normalize_bbox(box, width, height) for box in boxes]
            token_boxes = []
            tokenized_words = []
            for word, box in zip(texts, normalized_boxes):
                word_tokens = self.tokenizer.tokenize(word)
                tokenized_words.extend(word_tokens)
                token_boxes.extend([box] * len(word_tokens))
            
            # Tokenize the words
            encoded = self.tokenizer(tokenized_words, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
            
            # Add special tokens' boxes
            token_boxes = [[0, 0, 0, 0]] + token_boxes + [[0, 0, 0, 0]]

            # Ensure token_boxes length matches input_ids length
            if len(token_boxes) < self.max_length:
                token_boxes += [[0, 0, 0, 0]] * (self.max_length - len(token_boxes))
            else:
                token_boxes = token_boxes[:self.max_length]

            encoded['bbox'] = torch.tensor(token_boxes, dtype=torch.long)
            encoded['labels'] = torch.tensor([self.labels.get(label, -100) for label in actual_labels] + [-100] * (self.max_length - len(actual_labels)), dtype=torch.long)

            print(f"Input IDs length: {len(encoded['input_ids'][0])}")
            print(f"Bounding Boxes length: {len(encoded['bbox'])}")
            print(f"Attention Mask length: {len(encoded['attention_mask'][0])}")
            print(f"Labels length: {len(encoded['labels'])}")

        except Exception as e:
            print(f"Error during processing item {idx}: {e}")
            return None

        return encoded


    def normalize_bbox(self, bbox, width, height):
        """Normalize bounding box coordinates."""
        return [
            int(1000 * (bbox[0] / width)),
            int(1000 * (bbox[1] / height)),
            int(1000 * (bbox[2] / width)),
            int(1000 * (bbox[3] / height)),
        ]

    def extract_data(self, item):
        texts = []
        boxes = []
        labels = []
        if 'form' not in item:
            raise ValueError("Expected 'form' key in item data")
        for annotation in item['form']:
            if 'words' not in annotation:
                continue  # Skip annotations without words
            for word in annotation['words']:
                texts.append(word['text'])
                boxes.append(word['box'])  # Assume boxes are already in the correct format
                labels.append(annotation['label'])
        return texts, boxes, labels













