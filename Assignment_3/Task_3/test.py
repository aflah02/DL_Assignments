import os
import numpy as np
from datasets import load_dataset
import evaluate, torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

test_ds = load_dataset('wmt16', 'de-en', split='test')

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

model_checkpoint = "t5-small-finetuned-en-to-de/checkpoint-2500"
prefix = "translate English to German: "
    
tokenizer = AutoTokenizer.from_pretrained("t5-small")

max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "de"

def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    decoded_preds, decoded_labels = eval_preds
    # if isinstance(preds, tuple):
    #     decoded_preds = preds[0]
    # decoded_preds = tokenizer.batch_decode(preds[0], skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    bleu_1 = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
    bleu_2 = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=2)
    rouge_l = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=["rougeL"])
    result = {"bleu_1": bleu_1["bleu"], "bleu_2": bleu_2["bleu"], "rouge_l": rouge_l["rougeL"]}

    result = {k: round(v, 4) for k, v in result.items()}
    return result

def generate_translations(model):
    num_beams = 5
    translations = []
    labels = []
    
    for batch in tqdm(tokenized_test_dataset):
        translated = model.generate(torch.Tensor([batch["input_ids"]]).to(torch.int64).cuda(), num_beams=num_beams, max_length=max_target_length)
        translations.extend(tokenizer.batch_decode(translated, skip_special_tokens=True))
        labels.extend(tokenizer.batch_decode([batch["labels"]], skip_special_tokens=True))
    return translations, labels

tokenized_test_dataset = test_ds.map(preprocess_function, batched=True, remove_columns=["translation"])
# labels = tokenized_test_dataset["labels"]

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model.cuda()

eval_preds = generate_translations(model)

result = compute_metrics(eval_preds)
print(result)