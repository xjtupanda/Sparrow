import numpy as np
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import json
import os
import random

def read_jsonl(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def extract_data(path):
    data = read_jsonl(path)
    instruction_list = []
    for sample in data:
        instruction_list.append(sample['conversations'][0]['value'].replace('<video>', '').strip())
    return instruction_list

json_base = "/data/pandayin/data/data/visualize_jsonl"

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')

all_json_list = ["longalpaca_chat_fake_vid.jsonl", "longqlora_chat_fake_vid.jsonl", "sharegemini_webvid_core100k-clean.jsonl", "video-chatgpt-clean.jsonl"]
label_list = ['LongAlpaca', 'LongQLoRA', 'ShareGemini', 'Video-ChatGPT']


all_instruction_embedding_list = []
all_label_list = []

for label_name, json_name in zip(label_list, all_json_list):
    json_path = os.path.join(json_base, json_name)
    cur_json_data = read_jsonl(json_path)
    cur_json_data = random.sample(cur_json_data, 5000)
    instruction_list = [sample['conversations'][0]['value'].replace('<video>', '').strip() for sample in cur_json_data] # (D,)
    cur_instruction_embeddings = model.encode(instruction_list) # (N, D=768)
    all_instruction_embedding_list.append(cur_instruction_embeddings)

    cur_labels = [label_name] * len(instruction_list)
    all_label_list.extend(cur_labels)

all_instruction_embedding = np.concatenate(all_instruction_embedding_list, axis=0)

tsne = TSNE(n_components=2, verbose=1, init='pca', metric='cosine')
embeddings_2d = tsne.fit_transform(all_instruction_embedding)


np.save("video-ins-text-20k-embedding.npy", embeddings_2d)
