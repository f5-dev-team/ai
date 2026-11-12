from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.training.data.dataset import (
    load_records,
    prepare_records,
    split_records,
)
from src.training.data.labels import (
    LABELS,
    ID_TO_LABEL,
    LABEL_TO_ID,
)
from src.training.evaluate import compute_metrics


MODEL_NAME = "klue/roberta-base"

DATASET_PATH = Path(
    "data/training/dataset_v0.jsonl"
)

OUTPUT_DIR = Path(
    "artifacts/models/scholarship-classifier-v0"
)

MAX_LENGTH = 128


def tokenize_dataset(
        dataset: Dataset,
        tokenizer,
) -> Dataset:
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=[
            "text",
            "group_id",
        ],
    )


def main():
    records = load_records(DATASET_PATH)
    records = prepare_records(records)

    train_records, validation_records, test_records = (
        split_records(records)
    )

    print(f"Train: {len(train_records)}")
    print(f"Validation: {len(validation_records)}")
    print(f"Test: {len(test_records)}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        problem_type="multi_label_classification",
    )

    train_dataset = tokenize_dataset(
        Dataset.from_list(train_records),
        tokenizer,
    )

    validation_dataset = tokenize_dataset(
        Dataset.from_list(validation_records),
        tokenizer,
    )

    test_dataset = tokenize_dataset(
        Dataset.from_list(test_records),
        tokenizer,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=1,
        save_total_limit=1,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    trainer.train()

    print("\n=== Validation ===")
    print(trainer.evaluate())

    print("\n=== Test ===")
    print(
        trainer.evaluate(
            eval_dataset=test_dataset,
            metric_key_prefix="test",
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(
        f"\nModel saved: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()