import json

import numpy as np
from datasets import Dataset, DatasetDict
import pandas as pd
from ray import tune, train
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.text import SacreBLEUScore, Perplexity
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from utils import get_verse_pairs
from metrics import RhymeAccuracy


# Hyperparameters
MAX_LENGTH = 64
BATCH_SIZE = 80
GRACE_PERIOD = 3
MAX_EPOCHS = 30
N_TRIALS = 50
REDUCTION_FACTOR = 2
MODEL = "facebook/bart-base"
LEARNING_RATE = tune.qloguniform(1e-6, 1e-2, 1e-6)
WEIGHT_DECAY = tune.qloguniform(1e-6, 1e-2, 1e-6)
WARMUP_STEPS = tune.qrandint(0, 1000, 10)

tokenizer = AutoTokenizer.from_pretrained(MODEL)


def tokenizer_function(examples):
    inputs = examples["verse_a"]
    targets = examples["verse_b"]
    model_inputs = tokenizer(inputs, text_target=targets, truncation=True, max_length=MAX_LENGTH)

    return model_inputs


def model_init():
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    return model


def train_model(config):
    trial_id = train.get_context().get_trial_id()

    # Data preparation
    train_df = pd.read_csv("/pvc/data/processed/train.csv")
    dev_df = pd.read_csv("/pvc/data/processed/dev.csv")
    test_df = pd.read_csv("/pvc/data/processed/test.csv")

    data = {
        "train": Dataset.from_dict(get_verse_pairs(train_df, use_reverse=True)),
        "validation": Dataset.from_dict(get_verse_pairs(dev_df, use_reverse=False)),
        "test": Dataset.from_dict(get_verse_pairs(test_df, use_reverse=False))
    }
    data = DatasetDict(data)
    data = data.map(tokenizer_function, batched=True)

    model = model_init()
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Metrics
    bleu_1 = SacreBLEUScore(n_gram=1)
    bleu_2 = SacreBLEUScore(n_gram=2)
    bleu_3 = SacreBLEUScore(n_gram=3)
    bleu_4 = SacreBLEUScore(n_gram=4)
    perplexity = Perplexity().to("cuda")
    rhyme_accuracy = RhymeAccuracy()

    def compute_metrics(eval_pred, compute_result=False):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        logits = logits.to("cuda")
        labels = labels.to("cuda")

        preds = torch.argmax(logits, dim=-1)
        # Replace -100 in the labels as we can't decode them.
        labels = torch.where(labels != -100, labels, torch.tensor(tokenizer.pad_token_id))

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        perplexity.update(logits, labels)
        bleu_1.update(decoded_preds, decoded_labels)
        bleu_2.update(decoded_preds, decoded_labels)
        bleu_3.update(decoded_preds, decoded_labels)
        bleu_4.update(decoded_preds, decoded_labels)
        rhyme_accuracy.update(decoded_preds, decoded_labels)

        if compute_result:
            metrics = {
                "rhyme_accuracy": rhyme_accuracy.compute().item(),
                "perplexity": perplexity.compute().item(),
                "bleu_1": bleu_1.compute().item(),
                "bleu_2": bleu_2.compute().item(),
                "bleu_3": bleu_3.compute().item(),
                "bleu_4": bleu_4.compute().item(),
            }

            # Report metrics to Ray Tune for HPO
            train.report(metrics)

            # Reset metrics
            perplexity.reset()
            bleu_1.reset()
            bleu_2.reset()
            bleu_3.reset()
            bleu_4.reset()
            rhyme_accuracy.reset()

            return metrics

    training_args = TrainingArguments(
        output_dir=f"/pvc/checkpoints_bart/trial_{trial_id}",
        evaluation_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        eval_accumulation_steps=50,
        batch_eval_metrics=True,
        load_best_model_at_end=True,
        metric_for_best_model="rhyme_accuracy",
        save_strategy="epoch",
        num_train_epochs=MAX_EPOCHS,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        use_cpu=False,
        save_total_limit=1,
        logging_strategy="epoch",
    )

    trainer = Trainer(
        model=None,
        model_init=model_init,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Add metrics on test set to TensorBoard
    eval_results = trainer.evaluate(data["test"])
    logging_dir = training_args.logging_dir
    writer = SummaryWriter(log_dir=logging_dir)
    for key, value in eval_results.items():
        writer.add_scalar(f"test/{key}", value)
    writer.close()


if __name__ == "__main__":
    search_space = {
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_steps": WARMUP_STEPS,
    }

    hyperopt_search = HyperOptSearch(search_space, metric="rhyme_accuracy", mode="max")
    scheduler = AsyncHyperBandScheduler(
        grace_period=GRACE_PERIOD,
        max_t=MAX_EPOCHS,
        reduction_factor=REDUCTION_FACTOR,
    )

    analysis = tune.run(
        train_model,
        search_alg=hyperopt_search,
        scheduler=scheduler,
        num_samples=N_TRIALS,
        metric="rhyme_accuracy",
        mode="max",
        resources_per_trial={"cpu": 0.5, "gpu": 0.5},
        storage_path="/pvc/ray_results/bart",
    )

    print("Best hyperparameters found were: ", analysis.best_config)
    with open("/pvc/best_hyperparameters_bart.json", "w") as f:
        json.dump(analysis.best_config, f)
