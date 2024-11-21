import json
import os
import shutil
from tqdm import tqdm
import glob
from pathlib import Path

def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# for video-mme

# for category in Path('./').iterdir():
#     if not category.is_dir(): continue
#     for video_type in category.iterdir():
#         if not video_type.is_dir(): continue
#         video_dir = video_type.joinpath('video')
#         for video in video_dir.iterdir():
#             video_path = str(video)
#             save_path = os.path.join('extracted_frames', str(video_dir), video.stem)
#             os.makedirs(save_path, exist_ok=True)
            
#             extracted_command = f"ffmpeg -i {video_path} -vf fps=1 {save_path}/frame_%04d.png"
#             os.system(extracted_command)
           

# for share-gemini and video-chatgpt

video_file_list = glob.glob('./Activity_Videos/*', recursive=True)
for video_file in tqdm(video_file_list):
    video_path = os.path.basename(video_file)
    video_path, ext = os.path.splitext(video_path)
    
    save_path = os.path.join("extracted_frames/", video_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    extracted_command = f"ffmpeg -i {video_file} -vf fps=1 {save_path}/frame_%04d.png"
    os.system(extracted_command)
