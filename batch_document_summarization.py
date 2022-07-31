import json
import torch
from transformers import MBartTokenizer, MBartForConditionalGeneration
from datasets import load_dataset


def gen_batch(inputs, batch_size):
    batch_start = 0
    while batch_start < len(inputs):
        yield inputs[batch_start: batch_start + batch_size]
        batch_start += batch_size


def predict(
        model_name,
        input_records,
        output_file,
        max_source_tokens_count=600,
        batch_size=4
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = MBartTokenizer.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

    predictions = []
    for batch in gen_batch(input_records, batch_size):
        texts = [r["text"] for r in batch]
        input_ids = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_source_tokens_count
        )["input_ids"].to(device)

        output_ids = model.generate(
            input_ids=input_ids,
            no_repeat_ngram_size=4
        )
        summaries = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for s in summaries:
            print(s)
        predictions.extend(summaries)
    with open(output_file, "w") as w:
        for p in predictions:
            w.write(p.strip().replace("\n", " ") + "\n")


gazeta_test = load_dataset('IlyaGusev/gazeta')["test"]
predict("IlyaGusev/mbart_ru_sum_gazeta", list(gazeta_test), "mbart_predictions.txt")
