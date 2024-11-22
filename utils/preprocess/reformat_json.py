import json
import os
import numpy as np
from tqdm import tqdm
from glob import glob

def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """
    if total_num_frames == desired_num_frames:
        return [idx for idx in range(total_num_frames)]
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

def read_jsonl(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

base_dir = "/data/pandayin/data/video_data/extracted_frames"
json_data = read_jsonl("video-chatgpt.jsonl")

NUM_FRAMES=24
new_data = []
for item in tqdm(json_data):
    vid_file = item['video']
    vid_path = os.path.join(base_dir, os.path.splitext(vid_file)[0])
    video_frame_list = sorted(glob(os.path.join(vid_path, '*')))
    duration = len(video_frame_list)
    assert duration > 0, f"unavailable video_path: {vid_path}"
    selected_frame_ids = get_seq_frames(duration, min(duration, NUM_FRAMES))
    
    total_frames = len(selected_frame_ids)
    convs = item['conversations']
    convs[0]['value'] = convs[0]['value'].replace('<video>', '').strip()
    prefix_prompt = ""
    image_mapping_dict = {}
    for idx in range(total_frames):
        prefix_prompt += f"<image_{idx:02d}>" + '\n'
        image_mapping_dict[f"<image_{idx:02d}>"] = video_frame_list[selected_frame_ids[idx]]
    convs[0]['value'] = prefix_prompt + convs[0]['value']
    new_convs = []
    for idx, rou in enumerate(convs):
        if idx % 2 == 0:
            new_convs.append(
                {
                    'role': 'user',
                    'content': rou['value']
                }
            )
        else:
            new_convs.append(
                {
                    'role': 'assistant',
                    'content': rou['value']
                }
            )
        
    new_data.append(
        {
            'id': item['id'],
            'image': image_mapping_dict,
            'conversations': new_convs
        }
    )

write_json('./minicpm-videochatgpt-100k-24frames.json', new_data)

# minicpm-sharegemini-100k-24frames.json
# minicpm-videochatgpt-100k-24frames.json