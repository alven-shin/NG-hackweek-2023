# %%
SEED = 69420
DO_TRAINING = True
EPOCHS_TO_TRAIN = 5
BASE_MODEL = "google/vit-base-patch16-224"
OUTPUT_MODEL_DIR = "./vit-base-pcb"

# %%
from datasets import load_dataset
from datasets import DatasetDict

ds = load_dataset("./dataset")

ds_train_devtest = ds["train"].train_test_split(test_size=0.3, seed=SEED)
ds_devtest = ds_train_devtest["test"].train_test_split(test_size=0.5, seed=SEED)

dataset = DatasetDict(
    {
        "train": ds_train_devtest["train"],
        "eval": ds_devtest["train"],
        "test": ds_devtest["test"],
    }
)

dataset

# %%
from transformers import ViTImageProcessor

model_name = BASE_MODEL if DO_TRAINING else OUTPUT_MODEL_DIR
processor = ViTImageProcessor.from_pretrained(model_name)
# processor(dataset["train"][-1]["image"], return_tensors="pt")


# %%
def transform(batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x for x in batch["image"]], return_tensors="pt")

    # Don't forget to include the labels!
    inputs["label"] = batch["label"]
    return inputs


prepared_ds = dataset.with_transform(transform)
# prepared_ds["eval"][0]

# %%
import torch


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch]),
    }


# %%
import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")


def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
    )


# %%
from transformers import ViTForImageClassification

labels = dataset["train"].features["label"].names

model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes=True,
)


# %%
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=OUTPUT_MODEL_DIR,
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=EPOCHS_TO_TRAIN,
    # fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="tensorboard",
    load_best_model_at_end=True,
)

# %%
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["eval"],
    tokenizer=processor,
)

# %%
if DO_TRAINING:
    try:
        train_results = trainer.train(resume_from_checkpoint=True)
    except ValueError:
        train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

# %%
metrics = trainer.evaluate(prepared_ds["test"])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

# %%
from transformers import ViTImageProcessor, ViTForImageClassification

processor = ViTImageProcessor.from_pretrained(OUTPUT_MODEL_DIR)

model = ViTForImageClassification.from_pretrained(
    OUTPUT_MODEL_DIR,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes=True,
)

idx = 15
image = dataset["test"][idx]["image"]
actual_label = dataset["test"][idx]["label"]
inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
labels[predicted_label]

display(
    image, f"actual: {labels[actual_label]}", f"predicted: {labels[predicted_label]}"
)
