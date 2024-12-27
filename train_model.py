# from transformers import Trainer, TrainingArguments
# import torch
# from data_loader import FUNSDDataset
# from transformers import LayoutLMTokenizer

# class CustomDataCollator:
#     def __call__(self, features):
#         # Initialize batch dictionary with empty lists
#         batch = {
#             k: [] for k in ['input_ids', 'attention_mask', 'bbox', 'labels']
#         }

#         # Determine the maximum lengths in this batch
#         max_len_input_ids = max(len(f['input_ids'][0]) for f in features)
#         max_len_bbox = max(len(f['bbox']) for f in features)

#         # Pad sequences and append to batch lists
#         for f in features:
#             batch['input_ids'].append(self.pad_to_max_length(f['input_ids'][0], max_len_input_ids, pad_value=0, is_tensor=True))
#             batch['attention_mask'].append(self.pad_to_max_length(f['attention_mask'][0], max_len_input_ids, pad_value=0, is_tensor=True))
#             batch['bbox'].append(self.pad_to_max_length(f['bbox'], max_len_bbox, pad_value=[0, 0, 0, 0], is_tensor=False))
#             batch['labels'].append(self.pad_to_max_length(f['labels'], max_len_input_ids, pad_value=-100, is_tensor=True))

#         # Convert to tensors
#         batch = {k: torch.stack(v) for k, v in batch.items()}

#         return batch

#     def pad_to_max_length(self, arr, max_length, pad_value=0, is_tensor=False):
#         """Pad the array to the maximum sequence length in the batch."""
#         current_length = len(arr)
#         padding_length = max_length - current_length

#         if padding_length > 0:
#             if is_tensor:
#                 padding = torch.full((padding_length,) + arr.shape[1:], pad_value, dtype=arr.dtype)
#                 return torch.cat([arr, padding], dim=0)
#             else:
#                 padding = [pad_value] * padding_length
#                 return torch.tensor(arr.tolist() + padding, dtype=torch.long)

#         return arr


# def train_model(dataset, model):
#     training_args = TrainingArguments(
#         output_dir='./results',
#         num_train_epochs=3,
#         per_device_train_batch_size=4,
#         logging_dir='./logs',
#         logging_steps=10,
#     )
#     data_collator = CustomDataCollator()
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=dataset,
#         data_collator=data_collator
#     )
#     trainer.train()

# if __name__ == "__main__":
#     tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
#     labels_dict = {'header': 0, 'question': 1, 'answer': 2, 'other': 3}
#     data_dir = '/home/os/kaggle-dataset/layoutlm/dataset/training_data'
#     train_dataset = FUNSDDataset(tokenizer, labels_dict, data_dir)
#     from model_setup import get_model
#     num_labels = len(labels_dict)
#     model = get_model(num_labels=num_labels)
#     train_model(train_dataset, model)




from transformers import Trainer, TrainingArguments
import torch
from data_loader import FUNSDDataset
from transformers import LayoutLMTokenizer

class CustomDataCollator:
    def __call__(self, features):
        # Initialize batch dictionary with empty lists
        batch = {
            k: [] for k in ['input_ids', 'attention_mask', 'bbox', 'labels']
        }

        # Determine the maximum lengths in this batch
        max_len_input_ids = max(len(f['input_ids'][0]) for f in features)
        max_len_bbox = max(len(f['bbox']) for f in features)

        # Pad sequences and append to batch lists
        for f in features:
            batch['input_ids'].append(self.pad_to_max_length(f['input_ids'][0], max_len_input_ids, pad_value=0, is_tensor=True))
            batch['attention_mask'].append(self.pad_to_max_length(f['attention_mask'][0], max_len_input_ids, pad_value=0, is_tensor=True))
            batch['bbox'].append(self.pad_to_max_length(f['bbox'], max_len_bbox, pad_value=[0, 0, 0, 0], is_tensor=False))
            batch['labels'].append(self.pad_to_max_length(f['labels'], max_len_input_ids, pad_value=-100, is_tensor=True))

        # Convert to tensors
        batch = {k: torch.stack(v) for k, v in batch.items()}

        return batch

    def pad_to_max_length(self, arr, max_length, pad_value=0, is_tensor=False):
        """Pad the array to the maximum sequence length in the batch."""
        current_length = len(arr)
        padding_length = max_length - current_length

        if padding_length > 0:
            if is_tensor:
                padding = torch.full((padding_length,) + arr.shape[1:], pad_value, dtype=arr.dtype)
                return torch.cat([arr, padding], dim=0)
            else:
                padding = [pad_value] * padding_length
                return torch.tensor(arr.tolist() + padding, dtype=torch.long)

        return arr


def train_model(dataset, model,tokenizer):
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_dir='./logs',
        logging_steps=10,  # Log every 10 steps
        evaluation_strategy="no",  # Evaluate during training
        eval_steps=50,  # Evaluate every 50 steps
        save_steps=50,  # Save checkpoint every 50 steps
    )
    data_collator = CustomDataCollator()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model("./saved_model")  # Save the model
    tokenizer.save_pretrained("./saved_model")  # Save the tokenizer

if __name__ == "__main__":
    tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
    labels_dict = {'header': 0, 'question': 1, 'answer': 2, 'other': 3}
    data_dir = '/home/os/kaggle-dataset/layoutlm/dataset/training_data'
    train_dataset = FUNSDDataset(tokenizer, labels_dict, data_dir)
    from model_setup import get_model
    num_labels = len(labels_dict)
    model = get_model(num_labels=num_labels)
    train_model(train_dataset, model,tokenizer)







