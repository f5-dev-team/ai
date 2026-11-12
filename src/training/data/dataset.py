import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.training.data.labels import LABELS, LABEL_TO_ID


def load_records(file_path: str | Path) -> list[dict]:
    path = Path(file_path)

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def encode_labels(labels: list[str]) -> list[float]:
    encoded = np.zeros(len(LABELS), dtype=np.float32)

    for label in labels:
        encoded[LABEL_TO_ID[label]] = 1.0

    return encoded.tolist()


def prepare_records(records: list[dict]) -> list[dict]:
    return [
        {
            "text": record["text"],
            "labels": encode_labels(record["labels"]),
            "group_id": record["group_id"],
        }
        for record in records
    ]


def split_records(
        records: list[dict],
        seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    groups = [record["group_id"] for record in records]
    unique_groups = set(groups)

    if len(unique_groups) >= 3:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.30,
            random_state=seed,
        )

        train_idx, temp_idx = next(
            splitter.split(records, groups=groups)
        )

        train_records = [records[i] for i in train_idx]
        temp_records = [records[i] for i in temp_idx]

        temp_groups = [
            record["group_id"]
            for record in temp_records
        ]

        second_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.50,
            random_state=seed,
        )

        val_idx, test_idx = next(
            second_splitter.split(
                temp_records,
                groups=temp_groups,
            )
        )

        validation_records = [
            temp_records[i]
            for i in val_idx
        ]

        test_records = [
            temp_records[i]
            for i in test_idx
        ]

        return (
            train_records,
            validation_records,
            test_records,
        )

    train_records, temp_records = train_test_split(
        records,
        test_size=0.30,
        random_state=seed,
    )

    validation_records, test_records = train_test_split(
        temp_records,
        test_size=0.50,
        random_state=seed,
    )

    print(
        "[WARNING] Group 수가 부족하여 "
        "smoke-test용 random split을 사용합니다."
    )

    return (
        train_records,
        validation_records,
        test_records,
    )