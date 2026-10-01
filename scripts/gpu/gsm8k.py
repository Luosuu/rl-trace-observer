"""Write GSM8K train/test parquet files in VERL's RL dataset format.

python scripts/gpu/gsm8k.py OUTPUT_DIR
"""

import re
import sys
from pathlib import Path

from datasets import Dataset, load_dataset

INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def _rows(split):
    for index, row in enumerate(split):
        answer = re.search(r"#### (\-?[0-9\.\,]+)", row["answer"]).group(1).replace(",", "")
        yield {
            "data_source": "openai/gsm8k",
            "prompt": [{"role": "user", "content": f"{row['question']} {INSTRUCTION}"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"index": index},
        }


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("openai/gsm8k", "main")
    for split in ("train", "test"):
        rows = list(_rows(dataset[split]))
        Dataset.from_list(rows).to_parquet(output / f"{split}.parquet")
        print(f"{split}: {len(rows)} rows")


if __name__ == "__main__":
    main()
