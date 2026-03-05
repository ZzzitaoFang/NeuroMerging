import argparse
import sys
import os
import shutil
import logging
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import dispatch_model
from accelerate.utils import infer_auto_device_map

from model_merging_methods.merging_methods import MergingMethod
from utils.utils import set_random_seed, smart_tokenizer_and_embedding_resize
from inference import create_llm, test_CMMLU, test_gsm8k, test_hendrycks_math, test_human_eval, \
    test_mbpp
from utils.load_config import cache_dir

PT_MODEL_PATH = "path/to/pretrained/model/"

task_model_mapping_dict = {
    "chinese": "chinese-llama-2-7b",
    "math": "metamath-7b-v1.0",
    "code": "CodeLlama-7b-hf"
}
finetuned_model_backbone_mapping_dict = {
    "chinese-llama-2-7b": "llama-2-7b-hf",
    "metamath-7b-v1.0": "llama-2-7b-hf",
    "CodeLlama-7b-hf": "llama-2-7b-hf",
}

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def create_checker(get_neuronal_behavior=False, get_point_behavior=False):
    if sum([get_neuronal_behavior, get_point_behavior]) > 1:
        raise ValueError(f"Only one args can be True.")
    
    neuronal_behavior = [                
                'q_proj.weight',
                'k_proj.weight',
                'v_proj.weight',
                'o_proj.weight',
                'gate_proj.weight',
                'up_proj.weight',
                'down_proj.weight',
                'embed_tokens.weight',
                'lm_head.weight',
            ]
            
    point_behavior = [
        'input_layernorm.weight',
        'post_attention_layernorm.weight',
        'norm.weight',
    ]
    
    neuronal_behavior = set(neuronal_behavior)
    point_behavior = set(point_behavior)
    
    def checker(string):
        if any(item in string for item in neuronal_behavior):
            return "neuronal_behavior"
        elif any(item in string for item in point_behavior):
            return "point_behavior"
        else:
            raise ValueError(f"Invalid dict name with {string}")
    
    if get_neuronal_behavior:
        return neuronal_behavior
    elif get_point_behavior:
        return point_behavior
    
    return checker

def get_merge_performance(args: argparse.Namespace, finetuned_model_names: list, merge_task_names: list,
                          models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizers: list):
    logger.info(f"configuration is {args}")
    
    SAVE_MERGE_MODELS = "/path/to/save/merged/models/"
    SAVE_MERGE_MODELS += f"top{round((1-args.param_value_mask_rate)*100)}"
    
    try:
        torch.cuda.empty_cache()
        pretrained_model1 = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=os.path.join(PT_MODEL_PATH, args.pretrained_model_name),
            device_map="cpu",
        )
        pretrained_tokenizer1 = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(PT_MODEL_PATH, args.pretrained_model_name)
        )
        pretrained_model2 = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=os.path.join(PT_MODEL_PATH, args.pretrained_model_name),
            device_map="cpu",
        )
        pretrained_tokenizer2 = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(PT_MODEL_PATH, args.pretrained_model_name)
        )
    except:
        torch.cuda.empty_cache()
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.pretrained_model_name, cache_dir=cache_dir)
        device_map = infer_auto_device_map(pretrained_model, no_split_module_classes=["GPT2Block"])
        pretrained_model = dispatch_model(pretrained_model, device_map=device_map)
        pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name,
                                                             cache_dir=cache_dir)

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="[PAD]"),
        model=pretrained_model1,
        tokenizer=pretrained_tokenizer1,
    )
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="[PAD]"),
        model=pretrained_model2,
        tokenizer=pretrained_tokenizer2,
    )
    
    for finetuned_model, finetuned_tokenizer in zip(models_to_merge, tokenizers):
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            model=finetuned_model,
            tokenizer=finetuned_tokenizer,
        )

    set_random_seed(seed=0)

    import time
    start_time = time.time()
    
    with autocast():
        merged_model1, merged_model2 = merging_method.get_merged_model(
            merged_model=pretrained_model1,
            merged_model2=pretrained_model2,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=[],
            trainers=trainers,
            scaling_coefficient=args.scaling_coefficient,
            nums_fisher_examples=None,
            fisher_scaling_coefficients=None,
            normalize_fisher_weight=None,
            minimal_fisher_weight=None,
            nums_regmean_examples=None,
            reduce_non_diagonal_ratio=None,
            param_value_mask_rate=args.param_value_mask_rate,
            weight_format=args.weight_format,
            weight_mask_rates=args.weight_mask_rates,
            use_weight_rescale=args.use_weight_rescale,
            mask_strategy=args.mask_strategy,
            mask_apply_method=args.mask_apply_method,
            models_use_deepcopy=False,
        )
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Run time: {elapsed_time:.8f}s")

    save_chinese_model_path = save_math_model_path = save_code_model_path = None
    
    save_merge_models = SAVE_MERGE_MODELS
    
    save_merge_models = os.path.join(save_merge_models, "part1")
    if args.merge_chinese:
        save_chinese_model_path = f"{save_merge_models}/chinese_7b"
    if args.merge_math:
        save_math_model_path = f"{save_merge_models}/math_7b"
    # if args.merge_code:
    #     save_code_model_path = f"{SAVE_MERGE_MODELS}/{'_'.join(merge_task_names)}/code_7b/{args.save_model_name}"

    # since the tokenizers of different tasks are different, we need to save them (together with the model) separately
    save_model_paths = [save_chinese_model_path, save_math_model_path]
    for i, save_model_path in enumerate(save_model_paths):
        if save_model_path is not None:
            logger.info(f"saving models at {save_model_path}...")
            merged_model1.generation_config.do_sample = True
            merged_model1.save_pretrained(save_directory=save_model_path)
            tokenizers[i].save_pretrained(save_directory=save_model_path)
    logger.info(f"models are saved")
    
    save_merge_models = SAVE_MERGE_MODELS
    
    save_merge_models = os.path.join(save_merge_models, "part2")
    if args.merge_chinese:
        save_chinese_model_path = f"{save_merge_models}/chinese_7b"
    if args.merge_math:
        save_math_model_path = f"{save_merge_models}/math_7b"
    # if args.merge_code:
    #     save_code_model_path = f"{SAVE_MERGE_MODELS}/{'_'.join(merge_task_names)}/code_7b/{args.save_model_name}"

    # since the tokenizers of different tasks are different, we need to save them (together with the model) separately
    save_model_paths = [save_chinese_model_path, save_math_model_path]
    for i, save_model_path in enumerate(save_model_paths):
        if save_model_path is not None:
            logger.info(f"saving models at {save_model_path}...")
            merged_model2.generation_config.do_sample = True
            merged_model2.save_pretrained(save_directory=save_model_path)
            tokenizers[i].save_pretrained(save_directory=save_model_path)
    logger.info(f"models are saved")


parser = argparse.ArgumentParser("Interface for merging LLMs")
parser.add_argument("--merge_chinese", action="store_true", default=False, help="whether to merge chinese model")
parser.add_argument("--merge_math", action="store_true", default=False, help="whether to merge math model")
parser.add_argument("--merge_code", action="store_true", default=False, help="whether to merge code model")

parser.add_argument("--merging_method_name", 
                    type=str, 
                    default="average_merging",
                    help="name of the method to merge models",
                    choices=["average_merging", "task_arithmetic", "ties_merging", "mask_merging", "neuro_merging"])

parser.add_argument("--scaling_coefficient", type=float, default=1.0,
                    help="scaling coefficient to merge the task vector")
parser.add_argument("--weight_format", type=str, help="the format of weights to be masked", default="delta_weight",
                    choices=["finetuned_weight", "delta_weight"])
parser.add_argument("--weight_mask_rate", type=float, default=0.1, help="weight mask rate")
parser.add_argument("--use_weight_rescale", action="store_true", default=False,
                    help="whether to rescale the weight by 1 / (1 - weight_mask_rate)")
parser.add_argument("--mask_strategy", type=str, help="mask strategy", default="random",
                    choices=["random", "magnitude"])
parser.add_argument("--mask_apply_method", type=str, default="average_merging",
                    help="merging method that the mask strategy applies",
                    choices=["average_merging", "task_arithmetic", "ties_merging", "neuro_merging"])

parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--end_index', type=int, default=sys.maxsize)
parser.add_argument("--tensor_parallel_size", type=int, default=1, help="numbers of gpus to use")
parser.add_argument("--param_value_mask_rate", type=float, default=0.8, help="ties weight mask rate")

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit()

if __name__ == "__main__":
    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    assert sum([args.merge_chinese, args.merge_math, args.merge_code]) >= 2, "should merge two tasks at least!"
    finetuned_model_names = []
    merge_task_names = []
    for merge_flag, task_name in zip([args.merge_chinese, args.merge_math, args.merge_code],
                                     ["chinese", "math", "code"]):
        if merge_flag:
            finetuned_model_names.append(task_model_mapping_dict[task_name])
            merge_task_names.append(task_name)

    pretrained_model_names = [finetuned_model_backbone_mapping_dict[finetuned_model_name] for finetuned_model_name in
                              finetuned_model_names]
    assert len(set(pretrained_model_names)) == 1, "the backbone of all the finetuned models should be the same!"
    args.pretrained_model_name = pretrained_model_names[0]
    args.weight_mask_rates = [args.weight_mask_rate for _ in range(len(finetuned_model_names))]

    if args.merging_method_name == "average_merging":
        args.save_model_name = f"{args.merging_method_name}"
    elif args.merging_method_name == "task_arithmetic":
        args.save_model_name = f"{args.merging_method_name}_scaling_coefficient_{args.scaling_coefficient}"
    elif args.merging_method_name == "ties_merging":
        args.save_model_name = f"{args.merging_method_name}_scaling_coefficient_{args.scaling_coefficient}_param_value_mask_rate_{args.param_value_mask_rate}"
    elif args.merging_method_name == "neuro_merging":
        args.save_model_name = f"{args.merging_method_name}"
    else:
        assert args.merging_method_name == "mask_merging"
        if args.mask_apply_method == "average_merging":
            mask_apply_method_name = f"{args.mask_apply_method}"
        elif args.mask_apply_method == "task_arithmetic":
            mask_apply_method_name = f"{args.mask_apply_method}_scaling_coefficient_{args.scaling_coefficient}"
        elif args.mask_apply_method == "ties_merging":
            mask_apply_method_name = f"{args.mask_apply_method}_scaling_coefficient_{args.scaling_coefficient}_param_value_mask_rate_{args.param_value_mask_rate}"
        elif args.mask_apply_method == "neuro_merging":
            mask_apply_method_name = f"{args.mask_apply_method}"

        weight_mask_rates = [str(weight_mask_rate) for weight_mask_rate in args.weight_mask_rates]
        args.save_model_name = f"{args.merging_method_name}/{mask_apply_method_name}/mask_{'_'.join(weight_mask_rates)}_rescale_{args.use_weight_rescale}"

    save_merge_log_path = f"./save_merge_llm_logs/{'_'.join(merge_task_names)}/{args.save_model_name}"
    os.makedirs(save_merge_log_path, exist_ok=True)
    
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(f"{save_merge_log_path}/{str(time.time())}.log")
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

    run_start_time = time.time()
    logger.info(f"********** Run starts. **********")

    models_to_merge = []
    finetuned_tokenizers = []
    merging_method = MergingMethod(merging_method_name=args.merging_method_name)

    for finetuned_model_name in finetuned_model_names:
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=os.path.join(PT_MODEL_PATH, finetuned_model_name),
            device_map="cpu",
        )
        
        finetuned_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(PT_MODEL_PATH, finetuned_model_name)
        )
        models_to_merge.append(finetuned_model)
        finetuned_tokenizers.append(finetuned_tokenizer)


    get_merge_performance(
        args=args, 
        finetuned_model_names=finetuned_model_names, 
        merge_task_names=merge_task_names,
        models_to_merge=models_to_merge,
        tokenizers=finetuned_tokenizers,
        trainers=[None for _ in range(len(finetuned_model_names))], 
        logger=logger,
        merging_method=merging_method, 
    )

    sys.exit()