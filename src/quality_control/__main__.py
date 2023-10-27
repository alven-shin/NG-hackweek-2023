import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments


def main() -> None:
    model = AutoModelForSequenceClassification.from_pretrained("google/vit-base-patch16-224", num_labels=5)
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    metric = evaluate.load("accuracy")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=lambda eval: compute_metrics(metric, eval),
    )
    trainer.train()

def compute_metrics(metric, eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
    main()
