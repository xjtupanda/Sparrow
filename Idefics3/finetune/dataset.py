import logging
import re
from typing import Dict

import torch
from PIL import Image
from torch.utils.data import Dataset
import logging
import re


logger = logging.getLogger(__name__)

idefics2_chat_template = "{% for message in messages %}{{message['role'].capitalize()}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"

FAKE_IMAGE_TOKEN = "<fake_token_around_image>"
IMAGE_TOKEN = "<image>"
END_OF_UTTERANCE_TOKEN = "<end_of_utterance>"
IGNORE_INDEX = -100

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        processor,
        max_length=2048,
    ):
        super(SupervisedDataset, self).__init__()
        self.raw_data = raw_data
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if 'image' in self.raw_data[i]:
            if isinstance(self.raw_data[i]["image"], str):
                images_list = [Image.open(self.raw_data[i]["image"]).convert("RGB")]
            elif isinstance(self.raw_data[i]["image"], Dict):
                ### for multi-images input, the template for every image is <image_xx>, such as <image_00>, <image_01>
                images_list = [Image.open(img_path).convert("RGB") for img_name, img_path in self.raw_data[i]["image"].items()]
        else:
            images_list = []
        
        convs = self.raw_data[i]["conversations"]
        return images_list, convs

def remove_image_tags(text):
    return re.sub(r'<image_\d+>', '', text)

# modified from: https://github.com/merveenoyan/smol-vision/blob/main/Idefics_FT.ipynb
def data_collator(examples, processor, max_length=2048):
    texts = []
    images = []
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
    ]
    for example in examples:
        images_list, convs = example
        messages = []

        for idx, rou in enumerate(convs):
            assert rou['role'] in ['user', 'assistant'], f"Unavailable role:{rou['role']}, not in ['user', 'assistant']"
            if idx == 0:
                # add image tokens for the first round (user round).
                content = [{"type": "image"} for _ in range(len(images_list))]
            else:
                content = []
            
            cur_text = rou['content']
            cur_text = remove_image_tags(cur_text).strip()
            content += [{"type": "text", "text": cur_text}]
            
            messages.append(
                {
                    "role": rou['role'],
                    "content": content
                }
            )
        prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(prompt.strip())
        images.append(images_list)

    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch