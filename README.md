
<h1 align="center">
    <img src="https://s21.ax1x.com/2024/11/25/pAhrS9s.png" width="220"/>
<br> Sparrow: Data-Efficient Video Fine-tuning Scheme
</h1>

**TL;DR:** *We proposed a data augmentation method (synthesizing "video" samples from long QA text data) to enrich the instruction diversity of video data, which facilitates more efficient training with comparable performance.*

## ✨ Highlights

🤔 **Main findings:** The importance of instruction diversity in video fine-tuning and how to efficiently improve it.

- We observed a limited instruction diversity in datasets developed for Video-LLMs, which led to low learning efficiency (<ins>More details and findings are available in our paper</ins>).
- Since text data could be a rich and economical source, we leveraged these data in a format that was more consistent with video instruction data.
  
<p align="center">
    <img src="https://s21.ax1x.com/2024/11/25/pAhyPTU.png" width="75%" height="75%">
</p>


## 📖  Implementation & Examples
For those interested in the implementation details and some real synthetic examples of our paper:
- How to translate text into images? Check `t2vid.py`.
- Synthetic images. Check `samples/` directory of this repo.


## 🌻 Acknowledgement
- Long text instruction data: [LongAlpaca](https://huggingface.co/datasets/Yukang/LongAlpaca-12k) and [LongQLoRA](https://huggingface.co/datasets/YeungNLP/LongQLoRA-Dataset).



