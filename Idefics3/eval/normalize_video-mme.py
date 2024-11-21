import argparse
import os
import pandas as pd
import json

CATEGORY_DICT = {
"meishi": "美食",
"lvxing": "旅行",
"lanqiu": "篮球",
"tianwen": "天文",
"zuqiu": "足球",
"shengwu": "生物医学",
"wutaiju": "舞台剧",

"shishang": "时尚",
"caijing": "财经商业",
"keji": "科技数码",

"renwen": "人文历史",
"wenxue": "文学艺术",
"dili": "地理",
"xinwen": "新闻",
"jilupian": "纪录片",
"zongyi": "综艺",
"dianying": "电影剧集",
"mengchong": "萌宠",
"youxi": "游戏电竞",
"donghua": "动画",

"shenghuo": "生活",
"moshu": "魔术",
"zaji": "杂技特效",
"shougong": "手工教程",
"qita": "其他",

"falv": "法律",
"tianjing": "田径",
"richang": "日常",
"yundong": "运动",

"duoyuzhong": "多语种"
}


VIDEO_TYPE_DICT = {
"s": "短视频 <= 2 min", 
"m": "中视频 4-15 min", 
"l": "长视频 30-60 min"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=str)
    parser.add_argument("--output-file", required=True, type=str)

    args = parser.parse_args()

    answers_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    video_types = "s,m,l".split(",")
    result_dir = args.result_dir
    idx = 0

    for video_type in video_types:

        for category in CATEGORY_DICT.keys():

            if not os.path.exists(f"{result_dir}/{video_type}/{category}.csv"):
                print(f"{result_dir}/{video_type}/{category}.csv does not exist")
                continue

            cate_df = pd.read_csv(f"{result_dir}/{video_type}/{category}.csv")
            for (cate_id, cate_row) in cate_df.iterrows():
                for cur_qs, cur_pred, cur_gt in zip(cate_row[['问题一', '问题二', '问题三']], 
                                                    cate_row[["模型回答一", "模型回答二", "模型回答三"]], 
                                                    cate_row[["答案一", "答案二", "答案三"]]):
                    
                    ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_qs,
                                   "pred": cur_pred,
                                   "GT": cur_gt,
                                   "category": video_type
                                   }) + "\n")
                    ans_file.flush()
                    
                    idx += 1
    ans_file.close()
