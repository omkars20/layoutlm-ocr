# model_setup.py
from transformers import LayoutLMForTokenClassification

def get_model(num_labels):
    model = LayoutLMForTokenClassification.from_pretrained('microsoft/layoutlm-base-uncased', num_labels=num_labels)
    return model

