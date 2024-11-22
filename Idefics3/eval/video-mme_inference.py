import argparse
import torch
from glob import glob
import os
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import json
from tqdm import tqdm
import random
import os
import math
import cv2

import time

import numpy as np
import pysubs2
import pandas as pd


VIDEO_TYPE_DICT = {
"s": "短视频 <= 2 min", 
"m": "中视频 4-15 min", 
"l": "长视频 30-60 min"
}

CATEGORY_DICT = {
"meishi": "美食",
"lvxing": "旅行",
"shishang": "时尚",
"lanqiu": "篮球",
"caijing": "财经商业",
"keji": "科技数码",
"zuqiu": "足球",
"tianwen": "天文",
"shengwu": "生物医学",
"wutaiju": "舞台剧",
"falv": "法律",
"shenghuo": "生活",
"moshu": "魔术",
"zaji": "杂技特效",
"shougong": "手工教程",
"xinwen": "新闻",
"jilupian": "纪录片",
"zongyi": "综艺",
"dianying": "电影剧集",
"mengchong": "萌宠",
"youxi": "游戏电竞",
"donghua": "动画",
"renwen": "人文历史",
"wenxue": "文学艺术",
"dili": "地理",
"tianjing": "田径",
"richang": "日常",
"yundong": "运动",
"qita": "其他",
"duoyuzhong": "多语种"
}

REPONSIBLE_DICT = {
    "lyd": ["meishi", "lvxing", "lanqiu", "tianwen"],
    "jyg": ["zuqiu", "shengwu", "wutaiju"],
    "wzh": ["shishang", "caijing", "keji", "duoyuzhong"],
    "wzz": ["renwen", "wenxue", "dili", "qita"],
    "zcy": ["xinwen", "jilupian", "zongyi", "dianying"],
    "by": ["mengchong", "youxi", "donghua"],
    "dyh": ["shenghuo", "moshu", "zaji", "shougong"],
    "lfy": ["falv", "tianjing", "richang", "yundong"]
}



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--model-path", type=str, default="")

    parser.add_argument("--responsible_man", type=str, default="dyh", help="Category of the video")
    parser.add_argument("--num-frames", type=int, default=24, help="Maximum number of input video frames.")
    parser.add_argument("--video_type", type=str, default="m", help="Type of the video. Choose from ['s', m', 'l']")
    parser.add_argument("--video_dir", type=str, default="../yt-videos", help="Directory containing the videos")
    parser.add_argument("--categories", type=str, default="", help="categories")
    parser.add_argument("--output_path", type=str, default="", help="Output path")
    parser.add_argument("--use_subtitles", action='store_true', help="Use subtitles")
    return parser.parse_args()


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


def load_video_from_file(video_path, num_frames=32):
    cv2_vr = cv2.VideoCapture(video_path)
    duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cv2_vr.get(cv2.CAP_PROP_FPS))

    video_data = []
    count = 0
    selected_frame_ids = get_seq_frames(duration, num_frames)
    while cv2_vr.isOpened():
        success, frame = cv2_vr.read()
        if not success:
            break
        if count in selected_frame_ids:
            image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            video_data.append(image)
        count += 1
    cv2_vr.release()
    video_info = {
        "fps": fps,
        "duration": duration, 
        "num_frames": len(video_data),
        "selected_frame_ids": selected_frame_ids 
    }

    return video_data, video_info

def load_video(video_path, num_frames=32):
    fps = 1 

    video_frame_list = sorted(glob(os.path.join(video_path, "*.png")))
    frame_idx = get_seq_frames(len(video_frame_list), min(num_frames, len(video_frame_list)))
    
    selected_frames = [video_frame_list[idx] for idx in frame_idx]

    image = [Image.open(img_file).convert('RGB') for img_file in selected_frames]
    video_data = image

    video_info = {
        "fps": fps,
        "duration": len(video_frame_list),
        "num_frames": len(frame_idx),
        "selected_frame_ids": frame_idx
    }

    return video_data, video_info


def RUN_YOUR_MODEL(prompt, video, model, processor):
    # Create inputs
    content = [{"type": "image"} for _ in range(len(video))]
    content += [{"type": "text", "text": prompt}]
    messages = [
        {
            "role": "user",
            "content": content
        }    
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[video], return_tensors="pt")
    inputs = {k: v.to('cuda') for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=32)

    generated_texts = processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True, clean_up_tokenization_space=True)[0]
    return generated_texts

def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_path = args.model_path
    model = AutoModelForVision2Seq.from_pretrained(model_path, device_map='auto', low_cpu_mem_usage=True, _attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16).eval()
    processor = AutoProcessor.from_pretrained(model_path, do_image_splitting=False)

    video_types = args.video_type.split(",")
    video_dir = args.video_dir

    assert args.categories or args.responsible_man, "Please specify the categories"

    if args.categories is not None and args.categories != "":
        categories = args.categories.split(",")
    else:
        categories = []
        for man in args.responsible_man.split(","):
            categories += REPONSIBLE_DICT[man]
        #categories = REPONSIBLE_DICT[args.responsible_man]
        

    df = pd.read_csv(f'{video_dir}/qa_annotations.csv')
    df["模型回答一"] = None
    df["模型回答二"] = None
    df["模型回答三"] = None

    for video_type in video_types:

        for category in categories:
            output_dir = f'{args.output_path}/{video_type}' 

            if os.path.exists(f'{output_dir}/{category}.csv'):
                continue

            condition1 = (df["子任务"] == CATEGORY_DICT[category])
            condition2 = (df["时长类别"] == VIDEO_TYPE_DICT[video_type])
            filtered_rows = df[condition1 & condition2]

            for idx, row in filtered_rows.iterrows():

                indices = ["一", "二", "三"]

                # Get the video name and questions
                video_name = row["视频命名"]
                questions = [row[f"问题{_}"] for _ in indices]

                # load extracted frames in dir.

                # video_name, _ = os.path.splitext(video_name)
                # video_file_name = f"{video_dir}/extracted_frames/{video_name}"

                # original script. load from raw .mp4 video.
                #video_file_name = f"{video_dir}/{category}/{video_type}/video/{video_name}"
                
                subtitles_file_name = ""
                for sub in os.listdir(f"{video_dir}/{category}/{video_type}/subtitles"):
                    if video_name[:-4] in sub and sub.endswith(".srt"):
                        subtitles_file_name = f"{video_dir}/{category}/{video_type}/subtitles/{sub}"
                        break

                # # Check if the video file exists
                # if not os.path.exists(video_file_name):
                #     print(f"No {video_file_name}.")
                #     continue
                
                user_prompt = "Answer with the option's letter from the given choices directly."

                video_file_name = f"{video_dir}/{category}/{video_type}/video/{video_name}"
                if os.path.exists(video_file_name):
                    # We use video frames pre-extracted at FPS=1. If unavailable, extract from .mp4 on the fly.
                    try:
                        video_name, _ = os.path.splitext(video_name)
                        video_file_name = f"{video_dir}/extracted_frames/{category}/{video_type}/video/{video_name}"
                        video, video_info = load_video(video_file_name, args.num_frames)
                    except:
                        video_file_name = f"{video_dir}/{category}/{video_type}/video/{video_name}"
                        video, video_info = load_video_from_file(video_file_name, args.num_frames)
                else:
                    print(f"No {video_file_name}."); continue


                start_time = time.time()
                for id, question in zip(indices, questions):
                    # Generate answers to the questions

                    if args.use_subtitles and os.path.exists(subtitles_file_name):
                        subtitles = []
                        subs = pysubs2.load(subtitles_file_name, encoding="utf-8")

                        for selected_frame_id in video_info["selected_frame_ids"]:
                            sub_text = ""
                            for sub in subs:
                                cur_time = pysubs2.make_time(frames=selected_frame_id, fps=video_info["fps"])
                                if sub.start < cur_time and sub.end > cur_time:
                                    sub_text = sub.text.replace("\\N", " ")
                                    break
                            subtitles.append(sub_text)


                    if args.use_subtitles and os.path.exists(subtitles_file_name):
                        subtitle_prompt = "This video's subtitles are listed below: \n" + "\n".join(subtitles) + "\n"
                    else:
                        subtitle_prompt = ""

                    prompt = subtitle_prompt + question + user_prompt

                    outputs = RUN_YOUR_MODEL(prompt, video, model, processor)

                    row[f"模型回答{id}"] = outputs

                    df.iloc[idx] = row

                    print(f"{prompt}\nAnswer: {outputs}.\n")
                
                print(f"Time taken to generate answers: {time.time() - start_time:.2f} seconds\n")
            

            results = df[condition1 & condition2] 

            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            results.to_csv(f'{output_dir}/{category}.csv', index=False)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
