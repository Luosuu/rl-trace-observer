"""Build a tiny tokenizer, model and dataset offline for CPU PPO runs."""

import json
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

DATA_SOURCE = "rl_trace_observer/count"
# Qwen2-style ChatML: VERL's Qwen continuous-token builder requires <|im_end|>
# and a single-token newline, which a byte-level BPE provides.
_SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _corpus() -> list[str]:
    numbers = " ".join(str(number) for number in range(100))
    return [f"<|im_start|>user\ncount to {n}<|im_end|>\n<|im_start|>assistant\n{numbers}\n" for n in range(10)] * 4


def build_tokenizer(path: Path) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400, special_tokens=_SPECIAL_TOKENS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tokenizer.train_from_iterator(_corpus(), trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        chat_template=_CHAT_TEMPLATE,
    )
    fast.save_pretrained(path)
    return fast


def build_model(path: Path, tokenizer: PreTrainedTokenizerFast) -> None:
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    Qwen2ForCausalLM(config).save_pretrained(path)


def build_dataset(path: Path, rows: int) -> None:
    records = [
        {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": f"count to {index % 10 + 1}"}],
            "ability": "count",
            "reward_model": {"style": "rule", "ground_truth": str(index % 10 + 1)},
            "extra_info": {"index": index},
        }
        for index in range(rows)
    ]
    pd.DataFrame(records).to_parquet(path)


def build_assets(root: Path) -> dict[str, Path]:
    model_dir = root / "model"
    tokenizer = build_tokenizer(model_dir)
    build_model(model_dir, tokenizer)
    build_dataset(root / "train.parquet", rows=16)
    build_dataset(root / "val.parquet", rows=4)
    (root / "assets.json").write_text(json.dumps({"vocab_size": len(tokenizer)}))
    return {"model": model_dir, "train": root / "train.parquet", "val": root / "val.parquet"}
