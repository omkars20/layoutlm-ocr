# import torch
# from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
# from PIL import Image
# import json
# import pandas as pd

# # Load the model and tokenizer
# model_path = "microsoft/layoutlm-base-uncased"
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
#     labels = []
#     linkings = []

#     for annotation in annotations['form']:
#         label = annotation['label']
#         linking = annotation.get('linking', [])
#         if 'words' not in annotation:
#             continue  # Skip annotations without words
#         for word in annotation['words']:
#             texts.append(word['text'])
#             normalized_box = normalize_bbox(word['box'], width, height)
#             boxes.append(normalized_box)
#             labels.append(label)
#             linkings.append(linking)

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

#     return image, encoded, texts, labels, linkings

# # Function to make predictions
# def predict(image_path, annotations, model, tokenizer):
#     image, inputs, texts, labels, linkings = preprocess_input(image_path, annotations, tokenizer)
#     model.eval()
#     with torch.no_grad():
#         outputs = model(**inputs)

#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=2)

#     return predictions, inputs, texts, labels, linkings

# # Function to decode predictions and create DataFrame
# def decode_predictions(predictions, inputs, labels_dict, tokenizer, texts, labels, linkings):
#     decoded_predictions = []

#     for token, label_id, text, linking in zip(inputs['input_ids'][0], predictions[0], texts, linkings):
#         decoded_token = tokenizer.convert_ids_to_tokens(token.item())  # Convert single token ID to token
#         decoded_predictions.append((decoded_token, labels_dict[label_id.item()], text, linking))

#     df = pd.DataFrame(decoded_predictions, columns=['Token', 'Predicted Label', 'Text', 'Linking'])
#     return df

# # Function to create structured DataFrame
# def create_structured_df(df):
#     structured_data = []
#     header_text = ""
#     question_text = ""
#     answer_text = ""

#     for _, row in df.iterrows():
#         if row['Predicted Label'] == 'header':
#             if header_text or question_text or answer_text:
#                 structured_data.append({
#                     'Header': header_text,
#                     'Question': question_text.strip(),
#                     'Answer': answer_text.strip()
#                 })
#                 question_text = ""
#                 answer_text = ""
#             header_text = row['Text']
#         elif row['Predicted Label'] == 'question':
#             question_text += row['Text'] + ' '
#         elif row['Predicted Label'] == 'answer':
#             answer_text += row['Text'] + ' '

#     # Add the last accumulated entry
#     if header_text or question_text or answer_text:
#         structured_data.append({
#             'Header': header_text,
#             'Question': question_text.strip(),
#             'Answer': answer_text.strip()
#         })

#     structured_df = pd.DataFrame(structured_data)
#     return structured_df



# # Example usage
# if __name__ == "__main__":
#     # Replace with the actual path to your image
#     image_path = "dataset/testing_data/images/82092117.png"
    
#     # Load annotations from a JSON file
#     annotations_path = "dataset/testing_data/annotations/82092117.json"
#     with open(annotations_path, 'r') as file:
#         annotations = json.load(file)
    
#     predictions, inputs, texts, labels, linkings = predict(image_path, annotations, model, tokenizer)
#     df = decode_predictions(predictions, inputs, labels_dict, tokenizer, texts, labels, linkings)
    
#     structured_df = create_structured_df(df)
    
#     print(structured_df)
#     print("\nLabel counts:\n", df['Predicted Label'].value_counts())

#     # Save the structured DataFrame to a CSV file
#     structured_df.to_csv('structured_inference_output4.csv', index=False)




import cv2
from PIL import ImageDraw, ImageFont,Image
import json
def visualize_annotations(image_path, annotations):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for annotation in annotations['form']:
        box = annotation['box']
        label = annotation['label']
        text = annotation['text']

        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"{label}: {text}", fill="red")

    image.show()

# Example usage
image_path = "dataset/testing_data/images/82092117.png"
with open("dataset/testing_data/annotations/82092117.json") as f:
    annotations = json.load(f)

visualize_annotations(image_path, annotations)
