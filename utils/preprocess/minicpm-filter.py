import json
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

def read_jsonl(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def write_jsonl(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def count_image(conv):
    counter = 0 
    for rou in conv:
        counter += rou['value'].count("<image>")
    return counter

def count_video(conv):
    counter = 0 
    for rou in conv:
        counter += rou['value'].count("<video>")
    return counter

def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

path = "openbmb/MiniCPM-Llama3-V-2_5"
tokenizer = AutoTokenizer.from_pretrained(path, add_bos_token=True, add_eos_token=False, trust_remote_code=True)

# LongAlpaca-pics  LongQLoRA-pics
# longalpaca_chat_fake_vid.json  longqlora_chat_fake_vid.json
orig_data = read_json("longqlora_chat_fake_vid.json")
video_base_dir = "LongQLoRA-pics" 
new_data = []
print("Before filter sample number:", len(orig_data))

counter = []
for sample in tqdm(orig_data):
    video_dir = os.path.join(video_base_dir, os.path.splitext(sample['video'])[0])
    if not os.path.exists(video_dir): continue
    frame_count = len(os.listdir(video_dir))
    # filter according to frame length
    if frame_count > 90: continue
    # check video correspondence
    if (count_video(sample['conversations']) != 1) or ("<video>" not in sample['conversations'][0]['value']): continue

    conversations = '\n'.join([temp['value'] for temp in sample['conversations']])
    token_length = len(tokenizer(
                    conversations, padding=False, truncation=False,
                ).input_ids)
    if len(sample['conversations'][0]['value'].split(' ')) >= 30:  # 30 for LongQLoRA, 500 for LongAlpaca
        continue
    token_length += 96 * frame_count

    # filter according to token length
    if token_length > 8192: continue 
    sample['set'] = "LongQLoRA"
    sample['conversations'][0]['value'] = sample['conversations'][0]['value'].replace('<video>', '<image>')
    sample['image'] = [os.path.join(sample['video'], pic) for pic in sorted(os.listdir(video_dir))]
    del(sample['video'])
    new_data.append(sample)


print("After fiter sample number:", len(new_data))
write_json("minicpm-longqlora.json", new_data)
