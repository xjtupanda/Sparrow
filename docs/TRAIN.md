## Data Preparation

### Dataset Download
| Data file | Sample size |
|---|---|
| [VideoInstruct100K.json](https://huggingface.co/datasets/MBZUAI/VideoInstruct-100K/blob/main/VideoInstruct100K.json) | 100K |
| [sharegemini_webvid_core100k.json](https://huggingface.co/datasets/Share14/ShareGemini/blob/main/sharegemini_webvid_core100k.json) | 100K |
| [longalpaca_chat_fake_vid.json](https://huggingface.co/datasets/xjtupanda/T2Vid-Synthetic/blob/main/longalpaca_chat_fake_vid.json) | 9K |
| [longqlora_chat_fake_vid.json](https://huggingface.co/datasets/xjtupanda/T2Vid-Synthetic/blob/main/longqlora_chat_fake_vid.json) | 10K |

### Preprocess

1. Extract video frames at an FPS of 1.

Check [extract_video_frames.py](https://github.com/xjtupanda/T2Vid/blob/main/utils/preprocess/extract_video_frames.py).

2. Organize the format of training data.

We adopt the multi-image format of MiniCPM-V:
<details>
  <summary>
    <b>Multiple images data with 1 sample.</b>
  </summary>

```
  [
    {
      "image": {
        "<image_00>": "path/to/image_0.jpg",
        "<image_01>": "path/to/image_1.jpg",
        "<image_02>": "path/to/image_2.jpg",
        "<image_03>": "path/to/image_3.jpg"
      },
      "conversations": [
        {
          "role": "user", 
          "content": "How to create such text-only videos using CapCut?\n<image_00>\n<image_01>\n<image_02>\n<image_03>\n"
        }, 
        {
          "role": "assistant", 
          "content": "To create a text-only video as shown in the images, follow these steps in CapCut..."
        }
      ]
    }
  ]
```
</details>


We also provide a simple script of [format conversion](https://github.com/xjtupanda/T2Vid/blob/main/utils/preprocess/reformat_json.py) for reference.


## Training

Run the scripts

- `finetune_ds.sh` (30K mix data) 
- `finetune_ds-baseline.sh` (200K video data)

in `Idefics3/finetune/` and  `MiniCPM-V/finetune/`.

For example, to fine-tune Idefics3 on 30K mix data, run:
```Shell
cd Idefics3/finetune/
sh finetune_ds.sh
```
