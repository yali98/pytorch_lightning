import torch
from torch import nn
import numpy as np


from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel

from datasets import load_dataset
import evaluate
from transformers import TrainingArguments, Trainer
import wandb

wandb.init(project="Test")

data_files = {"train": "./train_data", "test": "./test_data"}
dataset=load_dataset("csv",data_files=data_files)    #결측치를 무조건 제거해야 함


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint="monologg/koelectra-base-v3-discriminator"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

print(len(tokenizer.vocab.keys()))

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)



def tokenize_function(examples):
    return tokenizer(examples["document"], padding="max_length", truncation=True)

print(dataset["test"][0])

tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset=tokenized_datasets["train"].shuffle(seed=42)
test_dataset=tokenized_datasets["test"].shuffle(seed=42)

training_args = TrainingArguments(output_dir="test_trainer",
                                  evaluation_strategy="epoch",
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=16,
                                  per_device_eval_batch_size=16,
                                  weight_decay=0.01,
                                  save_total_limit=3,
                                  num_train_epochs=10,
                                  fp16=True,)

print(len(train_dataset[3]["input_ids"]))



metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model.to(device),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,

)

trainer.train()