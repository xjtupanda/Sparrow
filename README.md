# T2Vid

*A data augmentation method to enrich the instruction diversity for finetuning Video-LLMs.*

## :sparkles: Highlights
:rocket: **Train less, achieve more:** Mixing in our data, one can achieve comparable or better performance, while the total sample size is only **15%**.
|  | Video-MME | MVBench | TempCompass |
|---|---|---|---|
| MiniCPM-V-2.5-8B<br><sub>zero-shot</sub> | 48.2 | 42.9 | 49.1 |
| MiniCPM-V-2.5-8B<br><sub>200K video data</sub> | 50.8 | 48.0 | 54.7 |
| **MiniCPM-V-2.5-8B<br><sub>20K video data +  10K our synthetic data</sub>** | **53.0** | **48.4** | **56.8** |
|  |  |  |  |
| Idefics3-8B<br><sub>zero-shot</sub> | 51.2 | 49.6 | 55.9 |
| Idefics3-8B<br><sub>200K video data</sub> | 53.3 | 50.7 | **62.9** |
| **Idefics3-8B<br><sub>20K video data +  10K our synthetic data</sub>** | **56.3** | **51.6** | 62.3 |
