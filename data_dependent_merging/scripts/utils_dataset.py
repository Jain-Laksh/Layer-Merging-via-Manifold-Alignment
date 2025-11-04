# aryan/scripts/utils_dataset.py
import os
import pandas as pd
import random

# Map our "task name" → actual MMLU subject list
MMLU_TASK_GROUPS = {
    "medical": ["clinical_knowledge", "professional_medicine"],
    "legal": ["professional_law", "international_law"],
    "math": ["high_school_mathematics", "college_mathematics"],
    "cs": ["computer_security", "machine_learning"],
    "humanities": ["world_religions", "philosophy"],
}


def load_mmlu_subject_csv(data_dir: str, subject: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "test", f"{subject}_test.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"MMLU file not found: {path}")
    return pd.read_csv(path)


def get_mmlu_samples(data_dir: str, task_name: str, num_samples: int):
    """
    Returns a list of dicts: {"text": prompt_string}
    """
    subjects = MMLU_TASK_GROUPS[task_name]
    rows = []

    for sub in subjects:
        df = load_mmlu_subject_csv(data_dir, sub)
        # shuffle indices
        idxs = list(range(len(df)))
        random.shuffle(idxs)
        for idx in idxs[: num_samples // len(subjects) + 1]:
            row = df.iloc[idx]
            # basic MMLU prompt: question + choices
            q = row["question"]
            choices = [row[c] for c in ["A", "B", "C", "D"] if c in row and not pd.isna(row[c])]
            prompt = q + "\n"
            for i, ch in zip(["A", "B", "C", "D"], choices):
                prompt += f"{i}. {ch}\n"
            rows.append({"text": prompt.strip()})
            if len(rows) >= num_samples:
                break
        if len(rows) >= num_samples:
            break

    return rows[:num_samples]
