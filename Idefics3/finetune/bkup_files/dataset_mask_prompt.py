import logging
import re
from typing import Dict
import torch
from PIL import Image
from torch.utils.data import Dataset
import logging
import re


'''
    This is a implementation of the vanilla data preparation.
    Specifics:
        1. The official script predicts both the instruction and the answer, while in common practice,
            only the loss for the answer part is calculated (instruction masked in loss calculation).
            To align with common practice, we implement this version, which only includes the loss for answer prediction.
        2. Since we observed a higher loss in the whole training process, we simply used the original implementation and deprecated this one.
'''
logger = logging.getLogger(__name__)

idefics3_chat_template = "<|begin_of_text|>{% for message in messages %}{{message['role'].capitalize()}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"

FAKE_IMAGE_TOKEN = "<fake_token_around_image>"
IMAGE_TOKEN = "<image>"
GLOBAL_IMAGE_TOKEN = "<global-img>"
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

def pad_sequence_custom(sequences, padding_side='right', padding_value=0):
    # Find the maximum length of the sequences
    max_len = max([seq.size(0) for seq in sequences])
    
    # Initialize a tensor with the padding value
    out_tensor = torch.full((len(sequences), max_len), padding_value, dtype=sequences[0].dtype)
    
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            out_tensor[i, :length] = seq  # Pad on the right
        elif padding_side == 'left':
            out_tensor[i, -length:] = seq  # Pad on the left
        else:
            raise ValueError("padding_side must be either 'left' or 'right'")
    
    return out_tensor

def data_collator(examples, processor, max_length=2048):
    image_str = f"{FAKE_IMAGE_TOKEN}{GLOBAL_IMAGE_TOKEN}{IMAGE_TOKEN * 169}{FAKE_IMAGE_TOKEN}"
    images = []
    
    batch_input_ids, batch_targets = [], []
    for example in examples:
        images_list, convs = example    # each sample contains a conversation list and (optionally) a image list.
        images.append(images_list)
        # Apply prompt templates
        input_id, target = [], []
        for i, rou in enumerate(convs):
            cur_role = rou['role'].capitalize()

            cur_role += ':' if i==0 else ': '
            if i == 0:
                cur_role = "<|begin_of_text|>"*2 + image_str * len(images_list) + cur_role
            
            # [1:] to remove the bos token "<|begin_of_text|>" 
            _input_id = processor.tokenizer(cur_role).input_ids[1:]
            _target = [IGNORE_INDEX] * len(_input_id)

            
            input_id += _input_id
            target += _target
            assert len(input_id) == len(target)

            cur_content = rou['content']
            cur_content = remove_image_tags(cur_content).strip()
            cur_content += "<end_of_utterance>\n"

            # [1:] to remove the bos token "<|begin_of_text|>" 
            _input_id = processor.tokenizer(cur_content).input_ids[1:]
            if rou['role']=='user':
                _target = [IGNORE_INDEX] * len(_input_id)
            elif rou['role']=='assistant':
                _target = _input_id
            else:
                raise NotImplementedError(f"Unvailable role:{rou['role']}, not in ['user','assistant']")

            input_id += _input_id
            target += _target
            assert len(input_id) == len(target)
        input_id = torch.tensor(input_id, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        batch_input_ids.append(input_id)
        batch_targets.append(target)

    
    batch_input_ids = pad_sequence_custom(
        batch_input_ids,
        padding_side=processor.tokenizer.padding_side,    # align with the default configuration..
        padding_value=processor.tokenizer.pad_token_id)
    
    batch_targets = pad_sequence_custom(
        batch_targets,
        padding_side=processor.tokenizer.padding_side,    # align with the default configuration..
        padding_value=IGNORE_INDEX)
    
    if processor.tokenizer.truncation_side == 'right':
        batch_input_ids = batch_input_ids[:, :max_length]
        batch_targets = batch_targets[:, :max_length]
    else:
        batch_input_ids = batch_input_ids[:, -max_length:]
        batch_targets = batch_targets[:, -max_length:]

    batch = processor(images=images, return_tensors="pt")

    batch["input_ids"] = batch_input_ids
    batch["labels"] = batch_targets
    batch.pop("rows", None)
    batch.pop("cols", None)
    return batch
