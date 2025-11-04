from datasets import load_dataset
import pandas as pd
import os

SUBJECTS = [
    'abstract_algebra', 'all', 'anatomy', 'astronomy',
    'business_ethics', 'clinical_knowledge', 'college_biology',
    'college_chemistry', 'college_computer_science', 'college_mathematics',
    'college_medicine', 'college_physics', 'computer_security',
    'conceptual_physics', 'econometrics', 'electrical_engineering',
    'elementary_mathematics', 'formal_logic', 'global_facts',
    'high_school_biology', 'high_school_chemistry',
    'high_school_computer_science', 'high_school_european_history',
    'high_school_geography', 'high_school_government_and_politics',
    'high_school_macroeconomics', 'high_school_mathematics',
    'high_school_microeconomics', 'high_school_physics',
    'high_school_psychology', 'high_school_statistics',
    'high_school_us_history', 'high_school_world_history', 'human_aging',
    'human_sexuality', 'international_law', 'jurisprudence',
    'logical_fallacies', 'machine_learning', 'management', 'marketing',
    'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
    'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
    'professional_law', 'professional_medicine', 'professional_psychology',
    'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
    'virology', 'world_religions'
]

root = os.path.expanduser("~/aryan/data/mmlu/test")
os.makedirs(root, exist_ok=True)

for subject in SUBJECTS:
    print(f"→ downloading {subject} ...")
    ds = load_dataset("cais/mmlu", subject)

    if "test" not in ds:
        print(f"   skipping {subject} (no test split)")
        continue

    test_split = ds["test"]
    df = pd.DataFrame({
        "question": test_split["question"],
        "A": [c[0] for c in test_split["choices"]],
        "B": [c[1] for c in test_split["choices"]],
        "C": [c[2] for c in test_split["choices"]],
        "D": [c[3] for c in test_split["choices"]],
        "answer": test_split["answer"],
    })
    out_path = os.path.join(root, f"{subject}_test.csv")
    df.to_csv(out_path, index=False)

print(f"✅ done. CSVs in {root}")
