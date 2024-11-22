import argparse
import os
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import json
import string
import copy
from typing import List, Dict
from tqdm import tqdm
import math
from torch.utils.data import Dataset, DataLoader
import re


PROMPT_TEMPLATE='''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]



def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def inference_with_llm(model, tokenizer, prompt):

    chat = PROMPT_TEMPLATE.format(sys_prompt = "You are an AI assistant who will help me to match an answer with several options of a single-choice question.",
                                  user_prompt = prompt
                                  )
    input_ids = tokenizer(chat, return_tensors="pt").input_ids.to("cuda")
    generated_ids = model.generate(input_ids, max_new_tokens=128, repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True)

    return response

def build_prompt(question, options, prediction):
    tmpl = (
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer.\n'
        'If the meaning of all options are significantly different from the answer, output Z. '
        'Your should directly output a single uppercase character, such as A, B, C, D (if they are valid options) and Z, and nothing else. Here are two examples.\n\n'
        'Example 1: \n'
        'Question: What is the main object in image?\n\nOptions: A. teddy bear.\nB. rabbit.\nC. cat.\nD. dog.\n\n'
        'Answer: a cute teddy bear\n\nOutput: A\n\n'
        'Example 2: \n'
        'Question: What is the main object in image?\n\nOptions: A. teddy bear.\nB. rabbit.\nC. cat.\nD. dog.\n\n'
        'Answer: Spider\n\nOutput: Z\n\n'
        'Now here is the question, options, and the answer, you should match and give me the option letter: \n'
        'Question: {}\n\nOptions: {}\n\nAnswer: {}\n\nOutput: '
    )
    return tmpl.format(question, options, prediction)

def can_infer_option(answer: str, choices: Dict[str, str]):
    # Choices is a dictionary
    if 'Failed to obtain answer via API' in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]
    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = copy.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            # 'A' might be a quantifier rather than a simple option. e.x., A cat.
            if 'A' in splits and len(splits) > 3:
                return False
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return False


def build_choices(option_str: str) -> Dict[str, str]:
    choice_list = option_str.split('\n')
    choice_list = [x.strip('.').strip() for x in choice_list]
    choice_dict = {}
    for option in choice_list:
        match = re.match(r"([A-Z])\.\s*(.*)", option)
        option_letter = match.group(1)
        option_text = match.group(2)
        choice_dict[option_letter] = option_text
    return choice_dict

def can_infer_text(answer: str, choices: Dict[str, str]):
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False

def can_infer(answer: str, choices: Dict[str, str]):
    '''
        Args:
            answer: String. model prediction.
            choices: Dict. Multi-choices to choose from. e.x. , {'A': 'cat', 'B': 'dog', 'C': 'bird'}
    '''
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)

def exact_match_and_llm(model, tokenizer, question, option_str, prediction):
    ''''
        Try exact matching first. If failed, try llm matching with at most 3 times.
    '''
    prediction = prediction.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
        "Answer:"
    ]
    for answer_prefix in answer_prefixes:
        prediction = prediction.replace(answer_prefix, "").strip()
    # first try exact matching, if failed, try llm matching.
    matches = re.search(r'[ABCD]\.', prediction)
    if matches is not None:
        return matches[0].strip('.')
    matches = re.search(r'[ABCD]', prediction)
    if matches is not None:
        return matches[0]

    prompt = build_prompt(question, option_str, prediction)
    choice_dict = build_choices(option_str)
    retry = 3
    while retry:
        try:
            response_message = inference_with_llm(model, tokenizer, prompt)
            ret = can_infer(response_message, choice_dict)
            if ret:
                return ret
            else:
                print(f'Output includes 0 / > 1 letter among candidates {set(choice_dict)} and Z: {response_message}')
            retry -= 1
        except Exception as e:
            print(f"Error: {e}")
    
    return 'Z'

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, num_frames=24):
        self.questions = questions
        self.num_frames = num_frames


    def __getitem__(self, index):
        line = self.questions[index]
        
        qs = line["prompt"]
        

        return qs

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    qs = batch
    
    return qs[0]


# DataLoader
def create_data_loader(questions, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader



def run_inference(args):
    # run inference with llms.
    model_path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2"
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    predictions = [json.loads(q) for q in open(os.path.expanduser(args.pred_file), "r")]
    predictions = get_chunk(predictions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")


    
    data_loader = create_data_loader(predictions, num_workers=0)

    for qs, line in tqdm(zip(data_loader, predictions), total=len(predictions)):
        idx = line["question_id"]
        cur_prompt = line["prompt"]
        # normalize the question.
        cur_prompt = cur_prompt.strip("Answer with the option's letter from the given choices directly.").strip("Question:").strip()
        # extract the question and the options.
        str_list = cur_prompt.split('A. ')
        #assert len(str_list) == 2
        if len(str_list) != 2:
            continue
        qs_str, option_str = str_list
        option_str = ('A. ' + option_str).strip()
        
        cur_pred = line['pred']
        extracted_ans = exact_match_and_llm(model, tokenizer, qs_str, option_str, cur_pred)

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "orig_pred": cur_pred,
                                   "pred": extracted_ans,
                                   "GT": line['GT'],
                                   "category": line['category']
                                   }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-3.1-8B-Instruct/", help="path to the judge llm.")
    parser.add_argument("--pred-file", type=str, required=True, help="The jsonl file containing the questions, answers and model predictions.")
    parser.add_argument("--output-file", type=str, required=True, help="The output jsonl file.")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    args = parser.parse_args()
    run_inference(args)
