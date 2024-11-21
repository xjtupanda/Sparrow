import os
import json
import argparse
import re


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=str)
    args = parser.parse_args()

    results = [json.loads(line) for line in open(args.result_file)]
    counter_dict = {}   # 'category': (num_correct, num_total)

    for item in results:
        cur_category, cur_answer, cur_GT = item['category'], item['pred'], item['GT']
        if cur_category not in counter_dict:
            counter_dict[cur_category] = [0, 0] # (num_correct, num_total)
        
        counter_dict[cur_category][0] += (cur_answer == cur_GT)
        counter_dict[cur_category][1] += 1

    
    # print res
    total_correct, total_sample = 0, 0
    for category, counter in counter_dict.items():
        total_correct += counter[0]
        total_sample += counter[1]
        print(f'Category: {category}, Score: {counter[0] / counter[1] * 100 :.2f}')
    
    print(f"Total Score: {total_correct / total_sample * 100 :.2f}")
