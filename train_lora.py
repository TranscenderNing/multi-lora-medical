from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import fire


model_id = "/data/ldn/llm-models/Meta-Llama-3.1-8B-Instruct"
dtype = "bfloat16"
device_map = "auto"
MAX_LENGTH = 512

per_device_train_batch_size = 1
gradient_accumulation_steps = 8
logging_steps = 10
num_train_epochs = 6

target_modules = ["q_proj", "v_proj"]
rank = 16

instruction_field = "question"

output_field = "answer"



def main(
    train_file="/data/ldn/medical-n-svf/data/5domains/train/骨科康复.json",
    output_dir="./outputs/骨科康复",
):
    ds = load_dataset("json", data_files=train_file, split="train")
    print("dataset", ds)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("tokenizer pad token", tokenizer.pad_token)
    print("tokenizer", tokenizer)

    def process_func(example):
        input_ids, attention_mask, labels = [], [], []
        # instruction = tokenizer(
        #     "\n".join(["Human: " + example[instruction_field]]).strip()
        #     + "\n\nAssistant: "
        # )
        instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n{example[instruction_field]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
        response = tokenizer(example[output_field] + tokenizer.eos_token)
        input_ids = instruction["input_ids"] + response["input_ids"]
        attention_mask = instruction["attention_mask"] + response["attention_mask"]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
    print("tokenized_ds", tokenized_ds)

    print("decoding input_ids")
    print(tokenizer.decode(tokenized_ds[1]["input_ids"]))
    print("decoding labels")
    print(
        tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[1]["labels"])))
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device_map
    )

    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=target_modules,
        lora_dropout=0.00,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print("config", config)

    model = get_peft_model(model, config)

    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        num_train_epochs=num_train_epochs,
    )
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")


if __name__ == "__main__":
    fire.Fire(main)
