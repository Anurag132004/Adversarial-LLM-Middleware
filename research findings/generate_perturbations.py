#!/usr/bin/env python3

# 1. Authentication
from huggingface_hub import login
import os, json, time, random
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Login to Hugging Face
HF_TOKEN = "hf_..."  # replace with your actual token
login(token=HF_TOKEN)

# 2. Configs
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
ADAPTER = "ShirinYamani/mistral7b-fine-tuned-qlora"
WORKDIR = "/kaggle/working"
INPUT_FILE = f"{WORKDIR}/outputs/mistral_output.jsonl"
OUTPUT_FILE = f"{WORKDIR}/outputs/mistral_output_with_perturbations.jsonl"
BATCH_SIZE = 8
SAVE_EVERY = 5
MAX_MEMORY = {0: "70GB"}

# 3. Load model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN,
    device_map="auto",
    max_memory=MAX_MEMORY,
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

# 4. Perturbation functions (no NLP used)
def character_level_perturbation(text, prob=0.2):
    homoglyphs = {'a': 'à', 'e': 'è', 'i': 'ì', 'o': 'ò', 'u': 'ù', 'c': 'ç'}
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < prob:
            if chars[i].lower() in homoglyphs:
                chars[i] = homoglyphs[chars[i].lower()]
            elif chars[i].isalpha():
                chars[i] = '' if random.random() < 0.5 else chars[i] * 2
    return ''.join(chars)

def word_level_perturbation(text, drop_prob=0.2, insert_prob=0.2):
    inserts = ['actually', 'basically', 'essentially']
    words, new_words = text.split(), []
    for word in words:
        if random.random() > drop_prob:
            new_words.append(word)
            if random.random() < insert_prob:
                new_words.append(random.choice(inserts))
    return ' '.join(new_words)

def semantic_level_perturbation(text):
    if " is " in text:
        return text.replace(" is ", " is NOT ", 1)
    return "What is NOT " + text.strip("?") + "?"

def perturb_question(q):
    mode = random.choices(["char", "word", "semantic"], weights=[0.5, 0.3, 0.2])[0]
    return (
        character_level_perturbation(q) if mode == "char" else
        word_level_perturbation(q) if mode == "word" else
        semantic_level_perturbation(q)
    )

# 5. Load input file
with open(INPUT_FILE, "r") as f:
    data = [json.loads(line) for line in f]

# 6. Generation + Save loop
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
buffer = []
for idx, item in enumerate(tqdm(data, desc="Processing")):
    pert_q = perturb_question(item["question"])
    prompt = f"### Human: {pert_q}\n### Assistant:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # logits data
    tok_data = []
    for step, tok in enumerate(gen_ids):
        scores = out.scores[step][0]
        logit = scores[tok].item()
        prob = F.softmax(scores, dim=-1)[tok].item()
        tok_data.append({
            "token": tokenizer.decode([tok]).strip(),
            "logit": round(logit, 4),
            "probability": round(prob, 6)
        })

    # combined output
    combined = {
        "question": item["question"],
        "ground_truth": item["ground_truth"],
        "response": item["answer"],
        "no_of_tokens_in_response": item["no_of_tokens"],
        "logits_data": item["logits_data"],
        "perturbated_question": pert_q,
        "perturbated_response": decoded,
        "no_of_tokens_in_perturbated_response": len(gen_ids),
        "perturbated_logits_data": tok_data
    }
    buffer.append(combined)

    if idx % SAVE_EVERY == 0 or idx == len(data) - 1:
        with open(OUTPUT_FILE, "a") as f:
            for rec in buffer:
                f.write(json.dumps(rec) + "\n")
        buffer.clear()
        torch.cuda.empty_cache()

print(f"\n✅ Done. Output saved to {OUTPUT_FILE}")
