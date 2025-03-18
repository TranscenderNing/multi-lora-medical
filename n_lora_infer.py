# -*- coding: utf-8 -*-


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import json


model_path = '/data/ldn/llm-models/Meta-Llama-3.1-8B-Instruct'
lora_path = '/data/ldn/medical-n-svf/outputs/dispatch/final_model' # lora权重路径
test_file = "/data/ldn/medical-n-svf/data/5domains/test/dispatch.json"
dispatch_result_file="dispatch_result.json"
result_file="result.json"

dtype = "bfloat16"
device_map = "auto"
batch_size = 8




test_model_dir = {
    "骨科康复": "/data/ldn/medical-n-svf/outputs/骨科康复/final_model",
    "脊髓损伤康复":"/data/ldn/medical-n-svf/outputs/脊髓损伤康复/final_model",
    "内科康复":"/data/ldn/medical-n-svf/outputs/内科康复/final_model",
    "言语、吞咽康复":"/data/ldn/medical-n-svf/outputs/言语、吞咽康复/final_model",
    "卒中康复":"/data/ldn/medical-n-svf/outputs/卒中康复/final_model",
}






def test_model(model, tokenizer):
    test_ds = load_dataset('json', data_files=test_file, split="train")

    def modify_question(example):
        example['question'] = f"<|start_header_id|>user<|end_header_id|>\n\n{example['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return example
    
    test_ds = test_ds.map(modify_question)
    print(test_ds[0])
    batched_test_ds = test_ds.batch(batch_size)
    
    correct_count = 0
    results = []
    print("anwser and generation")
    for batch in batched_test_ds:
        model_inputs = tokenizer(batch["question"], return_tensors="pt", padding="longest").to('cuda')
        generated_ids = model.generate(**model_inputs,max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for question, cls, raw_generation, answer in zip(batch["question"],batch["cls"], response,batch["answer"]):
            print('raw_generation',raw_generation)
            generation = raw_generation
            print(cls, generation)
            if generation == cls:
                correct_count += 1
            results += [
                    {
                        "question": question,
                        "answer": answer,
                        "cls": cls,
                        "cls_pred": generation
                    }
                ]
                

    print(f"Accuracy: {correct_count / len(test_ds)}")
    with open(dispatch_result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    return




def test_5_domain(model, tokenizer):
    # 根据dispatch模型的分类加载不同模型
    test_ds = load_dataset('json', data_files=dispatch_result_file, split="train")
    guke_ds = test_ds.filter(lambda x: "骨科康复" in x['cls_pred'])
    jisui_ds = test_ds.filter(lambda x: "脊髓损伤康复" in x['cls_pred'])
    neike_ds = test_ds.filter(lambda x: "内科康复" in x['cls_pred'])
    yanyu_ds = test_ds.filter(lambda x: "言语、吞咽康复" in x['cls_pred'])
    zuzhong_ds = test_ds.filter(lambda x: "卒中康复" in x['cls_pred'])
    total_size = len(guke_ds) + len(jisui_ds) + len(neike_ds) + len(yanyu_ds) + len(zuzhong_ds)
    all_ds = {
        "骨科康复": guke_ds,
        "脊髓损伤康复": jisui_ds,
        "内科康复": neike_ds,
        "言语、吞咽康复": yanyu_ds,
        "卒中康复": zuzhong_ds,
    }

    def modify_question(example):
        example['question'] = f"<|start_header_id|>user<|end_header_id|>\n\n{example['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return example

    
    for domain, path in test_model_dir.items():
        model.load_adapter(path, adapter_name=domain)
    
    print("load model done")
    print(model)    
    correct_count = 0
    results = []
    for domain, path in test_model_dir.items():
        print(f"Testing {domain}...")
        # load model
        print("set adaptor")
        model.set_adapter(domain)
        print(model)
        # load data
        test_ds = all_ds[domain]
        test_ds = test_ds.map(modify_question)
        print(test_ds[0])
        batched_test_ds = test_ds.batch(batch_size)
    
        results = []
        print("anwser and generation")
        for batch in batched_test_ds:
            model_inputs = tokenizer(batch["question"], return_tensors="pt", padding="longest").to('cuda')
            generated_ids = model.generate(**model_inputs,max_new_tokens=512)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for question, cls,cls_pred, raw_generation, answer in zip(batch["question"],batch["cls"],batch["cls_pred"], response,batch["answer"]):
                print('raw_generation',raw_generation)
                generation = raw_generation
                print(answer, generation)
                if generation == answer:
                    correct_count += 1
                results += [
                        {
                            "question": question,
                            "answer": answer,
                            "cls": cls,
                            "cls_pred": cls_pred,
                            "model_pred": generation
                        }
                    ]
                

    print(f"Accuracy: {correct_count / total_size}")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    return



if __name__ == "__main__":    
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("tokenizer pad token", tokenizer.pad_token, tokenizer.eos_token, tokenizer.padding_side)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path, torch_dtype=dtype, device_map=device_map
    )

    # first pass, dispatch model
    # lora_model = PeftModel.from_pretrained(model=model, model_id=lora_path, adapter_name="dispath_adaptor")
    # test_model(lora_model, tokenizer)
    
    
    test_5_domain(model, tokenizer)
    
    
    
    

