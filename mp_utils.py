import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from categories import name_en2zh, subcategories, categories
choices = ["A", "B", "C", "D"]

PATH_TO_DATA_CMMLU = "/path/to/data_cmmlu/"

category2subject = defaultdict(list)
for k,v in categories.items():
    for subject, subcat in subcategories.items():
        for c in subcat:
            if c in v:
                category2subject[k].append(subject)


def format_example(df, idx, subject, include_answer=True, cot=False):
    prompt_start = "题目："
    prompt = prompt_start + df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])

    # Chain-of-thought
    if cot:
        prompt += "\n逐步分析并给出答案选项。"
    else:
        prompt += "\n答案是："

    if include_answer:
        prompt += "{}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(dev_df, subject, prompt_end, num_few_shot=0, tokenizer=None, max_length=2048, cot=False):
    if cot: # Chain-of-thought
        prompt = "以下是关于{}的单项选择题，请分析并选出正确答案。\n\n".format(name_en2zh[subject])
    else:
        prompt = "以下是关于{}的单项选择题，请直接给出正确答案的选项。\n\n".format(name_en2zh[subject])

    # If no tokenizer, don't consider max length.
    if tokenizer==None:
        for i in range(num_few_shot):
            example = format_example(dev_df, i, subject)
            prompt += example
        return prompt + prompt_end

    start_end_token_len = len(tokenizer.encode(prompt)+tokenizer.encode(prompt_end))
    # If cannot fit in model even without training data, remove the prompt at the beginning.
    if start_end_token_len>max_length:
        return prompt_end

    prompt_list = []
    if num_few_shot > 0:
        for i in range(num_few_shot):
            example = format_example(dev_df, i, subject)
            prompt_list.append((example, tokenizer.encode(example)))

        while prompt_list != [] and sum(len(e[1]) for e in prompt_list) >= max_length - start_end_token_len:
            print(f"Warning: {len(prompt_list)} shot case exceeds max_input_length, remove 1 shot.")
            longest_length = max([len(e[1]) for e in prompt_list])
            prompt_list = [e for e in prompt_list if len(e[1]) != longest_length]
        for p in prompt_list:
            prompt += p[0]

    return prompt + prompt_end


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax


def run_eval(model, tokenizer, eval, args):

    if model:
        model.eval()

    subjects=sorted([f.split(".csv")[0] for f in os.listdir(os.path.join(PATH_TO_DATA_CMMLU, "test/"))])
    
    all_number = 0
    correct_number = 0
    for subject in subjects:
        
        dev_df = pd.read_csv(os.path.join(PATH_TO_DATA_CMMLU, "dev", subject + ".csv"), header=0, index_col=0)
        test_df = pd.read_csv(os.path.join(PATH_TO_DATA_CMMLU, "test", subject + ".csv"), header=0, index_col=0)

        acc, preds, confs , subject_number , subject_correct = eval(model=model,
                                 tokenizer=tokenizer,
                                 subject=subject,
                                 dev_df=dev_df,
                                 test_df=test_df,
                                 num_few_shot=5,
                                 max_length=2048,
                                 cot= False)
        all_number += subject_number
        correct_number += subject_correct



    acc = correct_number / all_number
    print("ALL Average accuracy {:.3f}".format(acc))
    return acc
