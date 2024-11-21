import argparse
import torch
from glob import glob
import os
import numpy as np
import cv2
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math
import random
from glob import glob

torch.manual_seed(1234)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)

    return seq

def load_video_from_file(video_path, num_frames=24):
    # input : xxx.mp4
    # output: a list of numpy array, each item corresponds to a frame.
    
    cv2_vr = cv2.VideoCapture(video_path)
    duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
    assert duration > 0, f"unavailable video_path: {video_path}" 
    video_data = []
    count = 0
    selected_frame_ids = get_seq_frames(duration, min(duration, num_frames))
    while cv2_vr.isOpened():
        success, frame = cv2_vr.read()
        if not success:
            break
        if count in selected_frame_ids:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            video_data.append(image)
        count += 1
    cv2_vr.release()
    
    return video_data


def load_video_from_dir(video_path, num_frames=24):
    # input : a dir that contains xxx.jpg/png
    # output: a list of Image array.
    frame_list = sorted(glob(os.path.join(video_path, '*')))
    duration = len(frame_list)
    assert duration > 0, f"unavailable video_path: {video_path}"
    selected_frame_ids = get_seq_frames(duration, min(duration, num_frames))
    selected_frame_list = [frame_list[idx] for idx in selected_frame_ids]
    selected_frame_list = [Image.open(frame).convert('RGB') for frame in selected_frame_list]

    video_data = selected_frame_list
    return video_data

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, video_folder, num_frames=24):
        self.questions = questions
        self.video_folder = video_folder
        self.num_frames = num_frames


    def __getitem__(self, index):
        line = self.questions[index]
        video_file = line["video"]
        video_file = os.path.join(self.video_folder, video_file)
        if os.path.isdir(video_file):
            video_data = load_video_from_dir(video_file, self.num_frames)
        else:
            video_data = load_video_from_file(video_file, self.num_frames)
        
        qs = line["prompt"]
        

        return qs, video_data

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    qs, videos = zip(*batch)
    
    return qs[0], videos[0]


# DataLoader
def create_data_loader(questions, video_folder, num_frames=24, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, video_folder, num_frames)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    model_path = os.path.expanduser(args.model_path)
    model = AutoModelForVision2Seq.from_pretrained(model_path, device_map='auto', low_cpu_mem_usage=True, _attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16).eval()
    processor = AutoProcessor.from_pretrained(model_path, do_image_splitting=False)
    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")


    data_loader = create_data_loader(questions, args.video_folder, num_frames=args.num_frames, num_workers=0)
    

    for (qs, videos), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["prompt"]

        # Create inputs
        content = [{"type": "image"} for _ in range(len(videos))]
        content += [{"type": "text", "text": cur_prompt}]
        messages = [
            {
                "role": "user",
                "content": content
            }    
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[videos], return_tensors="pt")
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

        generated_ids = model.generate(**inputs, max_new_tokens=32)

        generated_texts = processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True, clean_up_tokenization_space=True)[0]
    

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "pred": generated_texts,
                                   "GT": line['GT'],
                                   "category": line['category']
                                   }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    eval_model(args)
