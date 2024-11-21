# T2Vid: Efficient Video Finetuning Scheme for MLLMs
<p align="center">
        🤗 <a href="https://huggingface.co/datasets/xjtupanda/T2Vid-Synthetic">Dataset</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/pdf/2409.12191">Paper</a> &nbsp&nbsp  </a>
</p>

**TL;DR:** *We proposed a data augmentation method to enrich the instruction diversity of video data, which facilitates more efficient training without compromising performance.*

## :sparkles: Highlights
:rocket: **Train less, achieve more:** By mixing in our synthetic data, one can achieve comparable or better performance, while the total training sample size is only **15%**.
|  | Video-MME | MVBench | TempCompass |
|---|---|---|---|
| MiniCPM-V-2.5-8B<br><sub>zero-shot</sub> | 48.2 | 42.9 | 49.1 |
| MiniCPM-V-2.5-8B<br><sub>200K video data</sub> | 50.8 | 48.0 | 54.7 |
| **MiniCPM-V-2.5-8B<br><sub>20K video data +  10K synthetic data</sub>** | **53.0** | **48.4** | **56.8** |
|  |  |  |  |
| Idefics3-8B<br><sub>zero-shot</sub> | 51.2 | 49.6 | 55.9 |
| Idefics3-8B<br><sub>200K video data</sub> | 53.3 | 50.7 | **62.9** |
| **Idefics3-8B<br><sub>20K video data +  10K synthetic data</sub>** | **56.3** | **51.6** | 62.3 |
