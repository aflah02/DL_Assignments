#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 
train_ds = load_dataset('wmt16', 'de-en', split='train[:1%]')
val_ds = load_dataset('wmt16', 'de-en', split='validation')
test_ds = load_dataset('wmt16', 'de-en', split='test')

train_ds = train_ds.train_test_split(test_size = 0.5)["train"]

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

model_checkpoint = "t5-small"
prefix = "translate English to German: "
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "de"

def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    bleu_1 = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
    bleu_2 = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=2)
    rouge_l = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=["rougeL"])
    result = {"bleu_1": bleu_1["bleu"], "bleu_2": bleu_2["bleu"], "rouge_l": rouge_l["rougeL"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def train(model, model_checkpoint):
    batch_size = 16
    model_name = model_checkpoint.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-{source_lang}-to-{target_lang}",
        evaluation_strategy = "steps", 
        logging_strategy = "steps",
        save_strategy = "steps",
        learning_rate=2e-5,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=1,
        predict_with_generate=True,
        report_to = "wandb"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_datasets,
        eval_dataset=tokenized_dev_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

tokenized_train_datasets = train_ds.map(preprocess_function, batched=True)
tokenized_dev_datasets = val_ds.map(preprocess_function, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model.cuda()

train(model, model_checkpoint)