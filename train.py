from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)

training_args = TrainingArguments(output_dir="test_trainer")