# import torch
# from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
# from PIL import Image
# import json
# import pandas as pd

# # Load the model and tokenizer
# model_path = "./saved_model"
# model = LayoutLMForTokenClassification.from_pretrained(model_path)
# tokenizer = LayoutLMTokenizer.from_pretrained(model_path)

# # Define the label mapping
# labels_dict = {0: 'header', 1: 'question', 2: 'answer', 3: 'other'}

# # Function to normalize bounding boxes
# def normalize_bbox(bbox, width, height):
#     return [
#         int(1000 * (bbox[0] / width)),
#         int(1000 * (bbox[1] / height)),
#         int(1000 * (bbox[2] / width)),
#         int(1000 * (bbox[3] / height)),
#     ]

# # Function to preprocess the input
# def preprocess_input(image_path, annotations, tokenizer, max_length=512):
#     image = Image.open(image_path).convert("RGB")
#     width, height = image.size

#     texts = []
#     boxes = []
#     for annotation in annotations['form']:
#         if 'words' not in annotation:
#             continue  # Skip annotations without words
#         for word in annotation['words']:
#             texts.append(word['text'])
#             normalized_box = normalize_bbox(word['box'], width, height)
#             boxes.append(normalized_box)
#             print(f"Normalized box: {normalized_box}")

#     token_boxes = []
#     tokenized_words = []
#     for word, box in zip(texts, boxes):
#         word_tokens = tokenizer.tokenize(word)
#         tokenized_words.extend(word_tokens)
#         token_boxes.extend([box] * len(word_tokens))

#     encoded = tokenizer(tokenized_words, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
#     token_boxes = [[0, 0, 0, 0]] + token_boxes + [[0, 0, 0, 0]]

#     if len(token_boxes) < max_length:
#         token_boxes += [[0, 0, 0, 0]] * (max_length - len(token_boxes))
#     else:
#         token_boxes = token_boxes[:max_length]

#     encoded['bbox'] = torch.tensor(token_boxes, dtype=torch.long).unsqueeze(0)  # Add batch dimension

#     print(f"Final token_boxes length: {len(token_boxes)}")
#     print(f"Encoded bbox shape: {encoded['bbox'].shape}")

#     return image, encoded

# # Function to make predictions
# def predict(image_path, annotations, model, tokenizer):
#     image, inputs = preprocess_input(image_path, annotations, tokenizer)
#     model.eval()
#     with torch.no_grad():
#         outputs = model(**inputs)

#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=2)
#     return predictions, inputs

# # Decode predictions to labels
# def decode_predictions(predictions, inputs, labels_dict, tokenizer):
#     tokens = inputs['input_ids'].squeeze().tolist()
#     predicted_labels = predictions.squeeze().tolist()

#     decoded_predictions = []
#     for token, label_id in zip(tokens, predicted_labels):
#         if token != tokenizer.pad_token_id:  # Skip padding tokens
#             decoded_token = tokenizer.convert_ids_to_tokens(token)  # Convert single token ID to token
#             if decoded_token not in tokenizer.all_special_tokens:  # Skip special tokens
#                 label_id = label_id if isinstance(label_id, int) else label_id[0]  # Ensure label_id is an integer
#                 decoded_predictions.append((decoded_token, labels_dict.get(label_id, 'O')))
    
#     return decoded_predictions

# # Function to convert decoded predictions to a DataFrame
# def predictions_to_dataframe(decoded_predictions):
#     df = pd.DataFrame(decoded_predictions, columns=["Token", "Label"])
#     return df

# # Example usage
# if __name__ == "__main__":
#     # Replace with the actual path to your image
#     image_path = "dataset/testing_data/images/82092117.png"
    
#     # Load annotations from a JSON file
#     annotations_path = "dataset/testing_data/annotations/82092117.json"
#     with open(annotations_path, 'r') as file:
#         annotations = json.load(file)
    
#     predictions, inputs = predict(image_path, annotations, model, tokenizer)
#     decoded_predictions = decode_predictions(predictions, inputs, labels_dict, tokenizer)
#     df = predictions_to_dataframe(decoded_predictions)
#     print(df)
import torch
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
from PIL import Image, ImageDraw, ImageFont
import json
import pandas as pd

# Load the model and tokenizer
model_path = "./saved_model"
model = LayoutLMForTokenClassification.from_pretrained(model_path)
tokenizer = LayoutLMTokenizer.from_pretrained(model_path)

# Define the label mapping
labels_dict = {0: 'header', 1: 'question', 2: 'answer', 3: 'other'}

# Function to normalize bounding boxes
def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

# Function to preprocess the input
def preprocess_input(image_path, annotations, tokenizer, max_length=512):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    texts = []
    boxes = []
    for annotation in annotations['form']:
        if 'words' not in annotation:
            continue  # Skip annotations without words
        for word in annotation['words']:
            texts.append(word['text'])
            normalized_box = normalize_bbox(word['box'], width, height)
            boxes.append(normalized_box)
            print(f"Normalized box: {normalized_box}")

    token_boxes = []
    tokenized_words = []
    for word, box in zip(texts, boxes):
        word_tokens = tokenizer.tokenize(word)
        tokenized_words.extend(word_tokens)
        token_boxes.extend([box] * len(word_tokens))

    encoded = tokenizer(tokenized_words, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[0, 0, 0, 0]]

    if len(token_boxes) < max_length:
        token_boxes += [[0, 0, 0, 0]] * (max_length - len(token_boxes))
    else:
        token_boxes = token_boxes[:max_length]

    encoded['bbox'] = torch.tensor(token_boxes, dtype=torch.long).unsqueeze(0)  # Add batch dimension

    print(f"Final token_boxes length: {len(token_boxes)}")
    print(f"Encoded bbox shape: {encoded['bbox'].shape}")

    return image, encoded, token_boxes

# Function to make predictions
def predict(image_path, annotations, model, tokenizer):
    image, inputs, token_boxes = preprocess_input(image_path, annotations, tokenizer)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    return predictions, inputs, token_boxes, image

# Decode predictions to labels
def decode_predictions(predictions, inputs, labels_dict, tokenizer):
    tokens = inputs['input_ids'].squeeze().tolist()
    predicted_labels = predictions.squeeze().tolist()

    decoded_predictions = []
    for token, label_id in zip(tokens, predicted_labels):
        if token != tokenizer.pad_token_id:  # Skip padding tokens
            decoded_token = tokenizer.convert_ids_to_tokens(token)  # Convert single token ID to token
            if decoded_token not in tokenizer.all_special_tokens:  # Skip special tokens
                label_id = label_id if isinstance(label_id, int) else label_id[0]  # Ensure label_id is an integer
                decoded_predictions.append((decoded_token, labels_dict.get(label_id, 'O')))
    
    return decoded_predictions

# Function to convert decoded predictions to a DataFrame
def predictions_to_dataframe(decoded_predictions):
    df = pd.DataFrame(decoded_predictions, columns=["Token", "Label"])
    return df

# Function to draw bounding boxes with labels on the image
def draw_boxes(image, boxes, labels):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, label in zip(boxes, labels):
        if label == 'O':
            continue
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), label, fill="red", font=font)

    return image

# Example usage
if __name__ == "__main__":
    # Replace with the actual path to your image
    image_path = "dataset/testing_data/images/82092117.png"
    
    # Load annotations from a JSON file
    annotations_path = "dataset/testing_data/annotations/82092117.json"
    with open(annotations_path, 'r') as file:
        annotations = json.load(file)
    
    predictions, inputs, token_boxes, image = predict(image_path, annotations, model, tokenizer)
    decoded_predictions = decode_predictions(predictions, inputs, labels_dict, tokenizer)
    df = predictions_to_dataframe(decoded_predictions)

    # Only keep bounding boxes for tokens that are not special tokens or padding
    boxes_to_draw = [box for box, (token, label) in zip(token_boxes, decoded_predictions) if label != 'O']
    labels_to_draw = [label for token, label in decoded_predictions if label != 'O']

    # Draw boxes and labels on the image
    image_with_boxes = draw_boxes(image, boxes_to_draw, labels_to_draw)
    image_with_boxes.save("annotated_image.png")
    print("Annotated image saved as annotated_image.png")

    # Print the DataFrame
    print(df)
