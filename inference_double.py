import argparse
import jsonlines
import sys
import shutil
import logging
import os
import time
from tqdm import tqdm
import glob
import json
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from vllm import LLM, SamplingParams
from human_eval.human_eval.data import write_jsonl, read_problems, stream_jsonl
from transformers import LlamaForCausalLM, LlamaTokenizer
from model_merging_methods.mask_weights_utils import mask_model_weights
from utils.utils import set_random_seed, smart_tokenizer_and_embedding_resize
from utils.evaluate_llms_utils import batch_data, extract_answer_number, remove_boxed, last_boxed_only_string, process_results, \
    generate_instruction_following_task_prompt, get_math_task_prompt, generate_code_task_prompt, read_mbpp
from utils.load_config import cache_dir
from mp_utils import  run_eval
from hf_causal_model import eval


from typing import Optional
import os
# import tempfile
from datetime import datetime
import shutil
import numpy as np
import gc
from human_eval.human_eval.evaluate_functional_correctness import entry_point
from vllm.model_executor.parallel_utils import parallel_state
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel


finetuned_model_backbone_mapping_dict = {
    "chinese-llama-2-7b": "llama-2-7b-hf", # Can also change to your local path to model checkpoints
    "metamath-7b-v1.0": "llama-2-7b-hf",
    "CodeLlama-7b-hf": "llama-2-7b-hf",
}

PT_MODEL_PATH = "/path/to/PT/"
FT_MODEL_PATH = "./merged_models"

def recover_from_pretrained_model_double(
    topk, 
    pretrained_model_name, 
    args, 
    logger: logging.Logger, 
    lam: float,
    tmp_model_save_path,
    lam1,
    DARE,
):
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=os.path.join(PT_MODEL_PATH, pretrained_model_name), device_map="cpu"
    )
    finetuned_model1 = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=os.path.join(FT_MODEL_PATH, f"{topk}_double{'_DARE' + DARE if DARE else ''}", "part1", "math_7b"), device_map="cpu"
    )
    finetuned_model2 = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=os.path.join(FT_MODEL_PATH, f"{topk}_double{'_DARE' + DARE if DARE else ''}", "part2", "math_7b"), device_map="cpu"
    )
    pretrained_tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=os.path.join(PT_MODEL_PATH, pretrained_model_name),
    )
    finetuned_tokenizer1 = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=os.path.join(FT_MODEL_PATH, f"{topk}_double{'_DARE' + DARE if DARE else ''}", "part1", "math_7b"),
    )
    finetuned_tokenizer2 = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=os.path.join(FT_MODEL_PATH, f"{topk}_double{'_DARE' + DARE if DARE else ''}", "part2", "math_7b"),
    )
    
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="[PAD]"),
        model=pretrained_model,
        tokenizer=pretrained_tokenizer,
    )
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="[PAD]"),
        model=finetuned_model1,
        tokenizer=finetuned_tokenizer1,
    )
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="[PAD]"),
        model=finetuned_model2,
        tokenizer=finetuned_tokenizer2,
    )

    logger.info(f"Add with lambda={lam}...")
    pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
    finetuned_param_dict1 = {param_name: param_value for param_name, param_value in finetuned_model1.named_parameters()}
    finetuned_param_dict2 = {param_name: param_value for param_name, param_value in finetuned_model2.named_parameters()}

    with torch.no_grad():
        for param_name in finetuned_param_dict1.keys():
            pretrained_param_dict[param_name] = finetuned_param_dict1[param_name] * lam1 + finetuned_param_dict2[param_name] * lam + pretrained_param_dict[param_name]
    
    for param_name, param_value in finetuned_model1.named_parameters():
        param_value.data.copy_(pretrained_param_dict[param_name])
    
    # logger.info(f"saving recovered {finetuned_model_name} model at {tmp_model_save_path}...")
    os.makedirs(tmp_model_save_path, exist_ok=True)
    finetuned_model1.save_pretrained(save_directory=tmp_model_save_path)
    finetuned_tokenizer1.save_pretrained(save_directory=tmp_model_save_path)
    
    return None
    return pretrained_param_dict


def create_llm(tmp_model_save_path, tensor_parallel_size=1):
    return LLM(model=tmp_model_save_path, tensor_parallel_size=tensor_parallel_size,)


def test_CMMLU(tmp_model_save_path, args, logger: logging.Logger):
    tokenizer = LlamaTokenizer.from_pretrained(f"/path/to/chinese_7b") # Specific for Chinese_llama_alpaca
    model = LlamaForCausalLM.from_pretrained(
        tmp_model_save_path,
        torch_dtype=torch.float16, # Follow https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/inference_hf.py
        device_map="cuda",
    )
    acc = run_eval(model, tokenizer, eval, args)
    logger.info(f"ALL Average accuracy {acc}")
    
    del model, tokenizer
    gc.collect()

    return acc
    

def test_gsm8k(llm, test_data_path, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize):
    gsm8k_ins = []
    gsm8k_answers = []
    problem_prompt = get_math_task_prompt()
    logger.info(f"gsm8k test prompt is {problem_prompt}")
    with open(test_data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["question"])
            gsm8k_ins.append(temp_instr)
            temp_ans = item['answer'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start_index:end_index]
    gsm8k_answers = gsm8k_answers[start_index:end_index]
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=60)

    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=1024, stop=stop_tokens)
    logger.info(f"sampling params is {sampling_params}")

    res_completions = []
    for idx, prompt in enumerate(batch_gsm8k_ins):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    results = []
    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(gsm8k_ins, res_completions, gsm8k_answers)):
        y_pred = extract_answer_number(completion)
        if y_pred != None:
            results.append(float(y_pred) == float(prompt_answer))
        else:
            results.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    
    accuracy = sum(results) / len(results)
    logger.info(f"invalid outputs length is {len(invalid_outputs)}, invalid_outputs are {invalid_outputs}")
    logger.info(f"data index starts from {start_index}, ends at {end_index}")
    logger.info(f"gsm8k test data length is {len(results)}, accuracy is {accuracy}")
    logger.info(args)
    # if save_model_path is not None:
    #     shutil.rmtree(save_model_path, ignore_errors=True)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    return accuracy


def test_hendrycks_math(llm, test_data_path, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize, save_model_path=None):
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    problem_prompt = get_math_task_prompt()
    logger.info(f"MATH test prompt is {problem_prompt}")
    with open(test_data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["instruction"])
            hendrycks_math_ins.append(temp_instr)
            solution = item['output']
            temp_ans = remove_boxed(last_boxed_only_string(solution))
            hendrycks_math_answers.append(temp_ans)

    hendrycks_math_ins = hendrycks_math_ins[start_index:end_index]
    hendrycks_math_answers = hendrycks_math_answers[start_index:end_index]
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=50)

    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048, stop=stop_tokens)
    logger.info(f"sampling params is {sampling_params}")

    res_completions = []
    for idx, prompt in enumerate(batch_hendrycks_math_ins):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    results = []
    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)):
        res = process_results(prompt, completion, prompt_answer, invalid_outputs)
        results.append(res)
    accuracy = sum(results) / len(results)
    logger.info(f"invalid outputs length is {len(invalid_outputs)}, invalid_outputs are {invalid_outputs}")
    logger.info(f"data index starts from {start_index}, ends at {end_index}")
    logger.info(f"MATH test data length is {len(results)}, accuracy is {accuracy}")
    logger.info(args)
    if save_model_path is not None:
        shutil.rmtree(save_model_path, ignore_errors=True)

    del llm
    torch.cuda.empty_cache()


def test_human_eval(llm, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize, save_gen_results_folder=None):
    problems = read_problems()
    task_ids = sorted(problems.keys())[start_index: end_index]
    prompts = [problems[task_id]['prompt'] for task_id in task_ids]
    num_samples = len(prompts)
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048)

    shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)

    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = f"{save_gen_results_folder}/{args.start_index + i}.jsonl"

        prompt = prompts[i].replace('    ', '\t')
        prompt_batch = [generate_code_task_prompt(prompt)]

        ids_batch = [task_ids[i]]
        completion_seqs = []

        loops = 1

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

            with torch.no_grad():
                completions = llm.generate(prompt_batch, sampling_params)
            gen_seqs = [completions[0].outputs[0].text]

            if gen_seqs is not None:
                assert len(ids_batch) == 1
                task_id = ids_batch[0]

                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq.split("### Response:")[-1]
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen_seq.replace('\t', '    ')

                    completion_seqs.append(
                        {'task_id': task_id,
                         'completion': completion_seq,
                         'all_code': all_code,
                         }
                    )

        write_jsonl(output_file, completion_seqs)

    files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
    logger.info(f"find {len(files)} files in {save_gen_results_folder}")

    outputs = []
    for code_file in tqdm(files, total=len(files)):
        codes = [c for c in stream_jsonl(code_file)]
        for code in codes:
            completion = code['completion']
            completion = completion.replace("\r", "")
            completion = completion.strip()
            if '```python' in completion:
                logger.info("completion matches ```python")
                def_line = completion.index('```python')
                completion = completion[def_line:].strip()
                completion = completion.replace('```python', '')
                try:
                    next_line = completion.index('```')
                    completion = completion[:next_line].strip()
                except:
                    logger.info("wrong completion")
            if "__name__ == \"__main__\"" in completion:
                logger.info("completion matches __name__ == \"__main__\"")
                try:
                    next_line = completion.index('if __name__ == "__main__":')
                    completion = completion[:next_line].strip()
                except:
                    logger.info("wrong completion")
            if "# Example usage" in completion:
                logger.info("completion matches # Example usage")
                next_line = completion.index('# Example usage')
                completion = completion[:next_line].strip()
            # the following codes are used to deal with the outputs of code-alpaca
            if "The solution is:" in completion:
                logger.info("completion matches The solution is:")
                def_line = completion.index("The solution is:")
                completion = completion[def_line:].strip()
                completion = completion.replace('The solution is:', '')
                try:
                    next_line = completion.index('\n\nThe answer is:')
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("maybe wrong completion")
            if "The answer is:" in completion:
                logger.info("completion matches The answer is:")
                def_line = completion.index("The answer is:")
                completion = completion[def_line:].strip()
                completion = completion.replace('The answer is:', '')
                try:
                    next_line = completion.index('\n\nThe answer is:')
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("maybe wrong completion")
            code['completion'] = completion
        outputs += codes

    logger.info(f"save to {save_gen_results_folder}.jsonl")
    write_jsonl(f"{save_gen_results_folder}.jsonl", outputs)
    # if save_model_path is not None:
    #     shutil.rmtree(save_model_path, ignore_errors=True)

    del llm
    torch.cuda.empty_cache()
    
    return f"{save_gen_results_folder}.jsonl"


def test_mbpp(llm, test_data_path, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize, save_model_path=None, save_gen_results_folder=None):
    problems = read_mbpp(test_data_path)
    task_ids = sorted(problems.keys())[start_index: end_index]
    prompts = []
    for task_id in task_ids:
        prompt = f"\n{problems[task_id]['text']}\nTest examples:"
        if task_id == 493:
            # The test examples are too long, we choose to only include the function name.
            test_example = problems[task_id]['test_list'][0]
            prompt += f"\ncalculate_polygons(startx, starty, endx, endy, radius)"
        else:
            for test_example in problems[task_id]['test_list']:
                prompt += f"\n{test_example}"
        prompts.append(prompt)

    num_samples = len(prompts)
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048)

    shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)

    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = f"{save_gen_results_folder}/{args.start_index + i}.jsonl"

        prompt = prompts[i].replace('    ', '\t')
        prompt_batch = [generate_code_task_prompt(prompt)]

        ids_batch = [task_ids[i]]
        completion_seqs = []

        loops = 1

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

            with torch.no_grad():
                completions = llm.generate(prompt_batch, sampling_params)
            gen_seqs = [completions[0].outputs[0].text]

            if gen_seqs is not None:
                assert len(ids_batch) == 1
                task_id = ids_batch[0]

                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq.split("### Response:")[-1]
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen_seq.replace('\t', '    ')

                    completion_seqs.append(
                        {'task_id': task_id,
                         'completion': completion_seq,
                         'all_code': all_code,
                         }
                    )

        write_jsonl(output_file, completion_seqs)

    files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
    logger.info(f"find {len(files)} files in {save_gen_results_folder}")

    problems = read_mbpp(test_data_path)
    outputs = [[] for _ in range(len(problems))]
    for code_file in tqdm(files, total=len(files)):
        codes = [c for c in stream_jsonl(code_file)]
        for code in codes:
            task_id = code['task_id']
            completion = code['completion']
            completion = completion.strip()
            if '```python' in completion:
                logger.info("completion matches ```python")
                def_line = completion.index('```python')
                completion = completion[def_line:].strip()
                completion = completion.replace('```python', '')
                try:
                    next_line = completion.index('\n```')
                    completion = completion[:next_line].strip()
                except:
                    logger.info("wrong completion")
            if "__name__ == \"__main__\"" in completion:
                logger.info("completion matches __name__ == \"__main__\"")
                try:
                    next_line = completion.index('if __name__ == "__main__":')
                    completion = completion[:next_line].strip()
                except:
                    logger.info("wrong completion")
            if "# Example usage" in completion:
                logger.info("completion matches # Example usage")
                next_line = completion.index('# Example usage')
                completion = completion[:next_line].strip()
            if "# Test examples" in completion:
                logger.info("completion matches # Test examples")
                next_line = completion.index('# Test examples')
                completion = completion[:next_line].strip()
            # the following codes are used to deal with the outputs of code-alpaca
            if "The solution is:" in completion:
                logger.info("completion matches The solution is:")
                def_line = completion.index("The solution is:")
                completion = completion[def_line:].strip()
                completion = completion.replace('The solution is:', '')
                try:
                    next_line = completion.index('\n\nThe answer is:')
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("maybe wrong completion")
            if "The answer is:" in completion:
                logger.info("completion matches The answer is:")
                def_line = completion.index("The answer is:")
                completion = completion[def_line:].strip()
                completion = completion.replace('The answer is:', '')
                try:
                    next_line = completion.index('\n\nThe answer is:')
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("maybe wrong completion")
            outputs[task_id - 11].append(completion)

    logger.info(f"save to {save_gen_results_folder}.jsonl")
    with open(f"{save_gen_results_folder}.jsonl", "w", encoding="utf-8") as fout:
        json.dump(outputs, fout)
    if save_model_path is not None:
        shutil.rmtree(save_model_path, ignore_errors=True)

    del llm
    torch.cuda.empty_cache()


def resolve_lambda_code(lambda_code):
    eval = __builtins__.eval
    if type(lambda_code) is tuple:
        lambda_list = torch.tensor(lambda_code)
    elif isinstance(lambda_code, float) or isinstance(lambda_code, int):
        lambda_list = torch.tensor([lambda_code])
    elif "linear+" in lambda_code:  # 0.8+2.51+0.1
        _, start, end, step = lambda_code.split("+")
        lambda_list = np.arange(eval(start), eval(end), eval(step))
    elif "mergelist" in lambda_code:
        task_lambdas = lambda_code.split("+")[-1].split(",")
        lambda_list = np.array(task_lambdas).astype(float).tolist()
    else:
        raise NotImplementedError(f"Unable to decode lambda_code {lambda_code}")
    return lambda_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Interface for inference LLMs")
    # parser.add_argument("--finetuned_model_name", type=str, default="alpaca-7b", help="name of the finetuned language model",
    #                     choices=["chinese-llama-2-7b", "metamath-7b-v1.0","CodeLlama-7b-hf"])
    parser.add_argument("--exp_name", type=str, default="alpaca-7b", help="name of the exp_name")
    
    # parser.add_argument("--dataset_name", type=str, default="xnli", help="dataset to be used", choices=["cmmlu", "gsm8k", "human_eval"])
    parser.add_argument("--weight_format", type=str, help="the format of weights to be masked", 
                        default="delta_weight", choices=["finetuned_weight", "delta_weight"])
    parser.add_argument("--topk", type=str, default=None, help="topk")
    parser.add_argument("--lambda_code", type=str, default=None, help="lambda_code")
    
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=sys.maxsize)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)

    parser.add_argument("--weight_mask_rate", type=float, default=0.0, help="weight mask rate")
    parser.add_argument("--use_weight_rescale", action="store_true", default=False, help="whether to rescale the weight by 1 / (1 - weight_mask_rate)")
    parser.add_argument("--mask_strategy", type=str, help="mask strategy", default="random", choices=["random", "magnitude"])
    parser.add_argument("--wizardcoder_use_llama2_as_backbone", action="store_true", default=False, help="whether to use llama-2 as the backbone for WizardCoder")
    parser.add_argument("--test_our", action="store_true", default=False, help="whether to use llama-2 as the backbone for WizardCoder")
    parser.add_argument("--merge_model_name", type=str, default=None, help="name of the finetuned language model")
    parser.add_argument("--lam1", type=float, default=0.0)
    parser.add_argument("--DARE", type=str, default="")
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit()

    if args.weight_mask_rate == 0.0:
        save_model_name = f"{args.exp_name}" # top lam
        save_model_path = None
        just_inference = True

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    os.makedirs(f"./save_logs/{save_model_name}", exist_ok=True)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(f"./save_logs/{save_model_name}/{str(time.time())}.log")
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    if args.DARE:
        output_file = "./results/DARE_results.txt"
    else:
        output_file = "./results/results.txt"
    lambdas = resolve_lambda_code(args.lambda_code)
    logger.info(f"********** Run starts. **********")
    logger.info(f"configuration is {args}")

    for lam in lambdas:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        save_gen_results_folder = f"./save_gen_codes_results/{args.exp_name}/{timestamp}"
        
        if args.weight_format == "delta_weight":
            tmp_model_save_path = os.path.join(FT_MODEL_PATH, f"tmp_{timestamp}")
            os.makedirs(tmp_model_save_path)
            
            recover_from_pretrained_model_double(
                args.topk,
                pretrained_model_name='llama-2-7b-hf',
                args=args, 
                logger=logger, 
                tmp_model_save_path=tmp_model_save_path,
                lam=lam, 
                lam1=args.lam1,
                DARE=args.DARE,
            )

        acc_cmmlu = test_CMMLU(tmp_model_save_path, args=args, logger=logger)

        llm = create_llm(tmp_model_save_path, tensor_parallel_size=args.tensor_parallel_size,)
        
        acc_gsm8k = test_gsm8k(llm=llm, test_data_path="math_code_data/gsm8k_test.jsonl",
                            args=args, logger=logger,start_index=args.start_index, end_index=args.end_index)

        acc_human = entry_point(
            test_human_eval(llm=llm, args=args, logger=logger, start_index=args.start_index, 
                            end_index=args.end_index,save_gen_results_folder=save_gen_results_folder)
        )['pass@1']

        with open(output_file, "a+") as f:
            f.write(f"{args.topk}_{args.lam1}_{lam}_{args.DARE}:{acc_cmmlu},{acc_gsm8k},{acc_human}" + "\n")
        
        logger.info(f"Final Acc. with {args.topk} and lambda {lam}:cmmlu: {acc_cmmlu}\tgsm8k: {acc_gsm8k}\thuman_eval: {acc_human}")
        
        # with open(output_file, "a+") as f:
        #     f.write(f"{args.topk}_{args.lam1}_{lam}:{acc_gsm8k}" + "\n")
        
        # logger.info(f"Final Acc. with {args.topk} and lambda {lam}:\tgsm8k: {acc_gsm8k}")
        
        if os.path.exists(tmp_model_save_path):
            shutil.rmtree(tmp_model_save_path)
        
        del llm
        gc.collect()
        
        destroy_model_parallel()
        torch.cuda.empty_cache()
        print("===== DELETE =====")
        # if torch.distributed.is_initialized():
        #     torch.distributed.destroy_process_group()
        # parallel_state._DATA_PARALLEL_GROUP = None
        # parallel_state._MODEL_PARALLEL_GROUP = None
        # parallel_state._PIPELINE_PARALLEL_GROUP = None
        # parallel_state._TENSOR_MODEL_PARALLEL_GROUP = None
        # parallel_state._EMBEDDING_GROUP = None 
        # parallel_state._SHARDING_GROUP = None 