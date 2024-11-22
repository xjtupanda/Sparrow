## Data Preparation

### Benchmark Preparation

The benchmark files should be organized as such in `benchmarks`.

```Shell
benchmarks
├── video-mme
├── mvbench
│   ├── video
│   └── mvbench.jsonl
└── TempCompass
    ├── videos
    └── TempCompass.jsonl
```
We have provided processed jsonl files of MVBench and TempCompass for easier reproduction.
For full files (such as videos), please refer to the official guidelines to (apply) and download.

### Video-MME
1. Follow the [instruction](https://github.com/BradyFU/Video-MME?tab=readme-ov-file#-dataset) to apply for the benchmark.
2. (Optional) Extract video frames to speed up the evaluation process (I/O for long videos can be time-consuming), you may refer to:
https://github.com/xjtupanda/T2Vid/blob/main/utils/preprocess/extract_video_frames.py#L17-L30

### MVBench
1. Download the videos in [Link](https://huggingface.co/datasets/OpenGVLab/MVBench/tree/main/video).
2. Unzip all the files in the `video` folder.

### TempCompass
1. Download the videos at [tempcompass_videos.zip](https://huggingface.co/datasets/lmms-lab/TempCompass/blob/main/tempcompass_videos.zip).
2. Unzip the file and put all the videos in `videos` folder.

## Evaluation

Run the scripts

- `run_bench.sh` to evaluate on the three benchmarks.

in `Idefics3/eval/` and  `MiniCPM-V/eval/`.

Usage:
```Shell
cd Idefics3/eval/
sh run_bench.sh {exp_name} {CKPT_file_path} {NUM_FRAMES}
```
For example, running `zero-shot` inference with `Idefics-3`, using `24` frames:
```Shell
sh run_bench.sh zero-shot HuggingFaceM4/Idefics3-8B-Llama3 24
```