import os
import torch
import numpy as np
from mp_utils import choices, format_example, gen_prompt, softmax, run_eval



def eval(model, tokenizer, subject, dev_df, test_df, num_few_shot, max_length, cot):
    choice_ids = [tokenizer.convert_tokens_to_ids(choice) for choice in choices]
    cors = []
    all_conf = []
    all_preds = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        prompt_end = format_example(test_df, i, subject, include_answer=False)
        prompt = gen_prompt(dev_df=dev_df,
                            subject=subject,
                            prompt_end=prompt_end,
                            num_few_shot=num_few_shot,
                            tokenizer=tokenizer,
                            max_length=max_length)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        if "token_type_ids" in inputs: # For Falcon
            inputs.pop("token_type_ids")
        label = test_df.iloc[i, test_df.shape[1] - 1]

        with torch.no_grad():
            outputs = model(**inputs)
            last_token_logits = outputs.logits[:, -1, :]
            choice_logits = last_token_logits[:, choice_ids].detach().cpu().numpy()
            conf = softmax(choice_logits[0])[choices.index(label)]
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choice_logits[0])]

        all_preds += pred
        all_conf.append(conf)
        cors.append(pred == label)


    acc = np.mean(cors)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
   
    return acc, all_preds, all_conf,len(cors),cors.count(True)