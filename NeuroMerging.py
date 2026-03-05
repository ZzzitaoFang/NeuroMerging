import torch
from tqdm import tqdm


def create_checker(get_neuronal_behavior=False, get_point_behavior=False):
    if sum([get_neuronal_behavior, get_point_behavior]) > 1:
        raise ValueError(f"Only one args can be True.")
    
    neuronal_behavior = [                
        'q_proj.weight', # Please use the real weight_name in the model to replace
        'k_proj.weight', # List all the neuronal behavior weight names here
        'v_proj.weight', # E.g., q/k/v/o_proj.weight, up/down_proj.weight, FFN, etc.
    ]
    
    point_behavior = [
        'input_layernorm.weight', # Please use the real weight_name in the model to replace
        'norm.weight', # List all the point behavior weight names here
        'norm.bias', # E.g., 1D layernorm.weight, norm.weight, norm.bias, etc.
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


def resolve_sign(Tensor,):
    sign_to_mult = torch.sign(Tensor.sum(dim=0))
    majority_sign = torch.sign(sign_to_mult.sum())
    sign_to_mult[sign_to_mult == 0] = majority_sign
    
    return sign_to_mult


def disjoint_merge(Tensor, sign_to_mult): # [Top-k, merge_func, (Top-k's sign)]
    rows_to_keep = torch.where(
        sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
    )
    selected_entries = Tensor * rows_to_keep

    non_zero_counts = (selected_entries != 0).sum(dim=0).float()
    disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
        non_zero_counts, min=1
    )

    return disjoint_aggs


def ties_kernel(pt_w, ft_ws):
    original_shape = pt_w.shape
    
    ft_ws_flat = torch.stack([ft_w.view(-1) for ft_w in ft_ws], dim=0)
    
    final_signs = resolve_sign(ft_ws_flat)
    merged_tv = disjoint_merge(ft_ws_flat, final_signs)
    
    return merged_tv.view(original_shape)


def inner_ties_kernel(scales):   # scales (t=task, b=batch)
    sign = torch.sign(scales.sum(dim=0)) # (b,)
    majority_sign = torch.sign(sign.sum()) # c
    sign[sign == 0] = majority_sign
    
    keep_pos = torch.where( # (t, b)
        sign.unsqueeze(0) > 0, scales > 0, scales < 0
    )
    scales = scales * keep_pos # (t, b)
    
    non_zero_counts = (scales != 0).sum(dim=0).float() # (b,)
    scales = scales.sum(dim=0) / torch.clamp( # (b,)
        non_zero_counts, min=1
    )
    return scales # (b,)


def neuro_kernel(pt_w, ft_ws, key, cum_threshold=1): # K, rm, mf, lam
    if pt_w.ndim == 2:
        pt_w_ = pt_w.unsqueeze(0) # (1, m, n)
    elif pt_w.ndim == 1:
        pt_w_ = pt_w.unsqueeze(0).unsqueeze(0) # (1, 1, n)
        ft_ws = [ft_w.unsqueeze(0) for ft_w in ft_ws] # (n,) -> (1, n)
    else:
        raise KeyError(
            f"Expected {key} to be a 1D or 2D array, but got shape {pt_w.shape}")
        
    ft_ws = torch.stack(ft_ws, dim=0) # (k, m, n)
    dot_num = torch.sum(ft_ws * pt_w_, dim=2)  # (k, m)
    dot_den = torch.sum(pt_w_ * pt_w_, dim=2)  # (1, m)
    scale = dot_num / dot_den  # (k, m)
    
    proj = scale.unsqueeze(2) * pt_w_ # (k, m, n)
    perp = ft_ws - proj # (k, m, n)
    
    # 1: projection part (scale -> ties)
    scale = inner_ties_kernel(scale) # (m, 1)
    nor_vec = pt_w_ / torch.norm(pt_w_, dim=2, keepdim=True) # (1, m, n)
    proj = scale.unsqueeze(1) * nor_vec.squeeze(0) # (m, n)
    proj = torch.zeros_like(proj) # Type2
    
    # 2: perpendicular part (perp)
    perp = perp.permute(1, 0, 2) # (m, k, n)
    _, S, VT = torch.linalg.svd(perp, full_matrices=False)
    # U: (m, k, k), S: (m, k), Vh: (m, k, n)
    # U: (m, k, r), S: (r, r), Vh: (m, r, n)  r = min(k, n)
    
    S_squared = S ** 2
    cum_var = torch.cumsum(S_squared, dim=1) / \
        torch.sum(S_squared, dim=1, keepdim=True)  # (m, r)
        
    rank = torch.minimum( # (m,) r<<n & r<<k  # Adaptive SVD
        (cum_var<cum_threshold).sum(dim=1)+1, 
        torch.tensor(VT.shape[-2], device=perp.device)
    )
    
    V = VT.permute(0, 2, 1) # (m, n, r)
    
    scales = torch.matmul(perp, V) # (m, k, r)

    scales_ = []
    for m in range(scales.size(0)):
        scales_.append(inner_ties_kernel(scales[m])) # (r,)
    scales_ = torch.stack(scales_, dim=0) # (m, r)
    
    # representative vectors
    rep_vecs = [] # (m, n)
    for m in torch.arange(scales_.size(0)): # m
        rep_vecs.append((scales_[m, :rank[m]] * V[m, :, :rank[m]]).sum(dim=1))
        # (rank[i],) * (n, rank[i]) --> (n,)
    rep_vecs = torch.stack(rep_vecs, dim=0) # (m, n)
    
    if len(pt_w.shape) == 2:
        merged_tv = proj + rep_vecs
    else:
        merged_tv = proj.squeeze(0) + rep_vecs.squeeze(0)
    
    return merged_tv


######### =========== Main Merging Body (Psudo-code) =========== #########
# 0. Perform a top-k operation on the delta weights, keeping only the largest k%.
# TBD based on your implementation / situation

# 1. Create a checker function to identify neuronal behavior and point behavior weights.
checker = create_checker()

merge_dict = {}
tqdm_ptm_check = tqdm(
    PRE_TRAINED_DICT.items(), # Replace PRE_TRAINED_DICT with the real pre-trained model dictionary.
    desc=f"Neuro kernel operation", 
    unit="weight matrix", 
    dynamic_ncols=True, 
    leave=False, 
)

# 2. For each weight in the model, 
# use the checker to determine its behavior type and do the corresponding merging operation.
for key, ptm_value in tqdm_ptm_check:
    ft_values = [ft_check[key] for ft_check in FT_CHECKS] # Replace FT_CHECKS with the list of fine-tuned model dictionaries.
    
    this_operator = checker(key)

    # 3. If it's a neuronal behavior weight, 
    # apply the neuro_kernel to merge the weights from different fine-tuned models.
    if this_operator == "neuronal_behavior":
        adj_weight = neuro_kernel(
            ptm_value, 
            ft_values,
            key, 
        )
    # 4. If it's a point behavior weight, 
    # apply the ties_kernel to merge the weights from different fine-tuned models.
    # The number of point behavior weights is much smaller than the number of neuronal behavior weights.
    elif this_operator == "point_behavior":
        # adj_weight = np.zeros_like(ptm_value)   # Non-operation for point
        adj_weight = ties_kernel(
            ptm_value, 
            ft_values, 
        )
    else:
        raise ValueError(f"Operator {this_operator} is not valid.")

    # 5. Update the merged model's weights with the merged weights obtained from steps 3 and 4.
    merge_dict[key] = adj_weight

merge_dict # Continue with the rest of the process, e.g., saving the merged model, eval, etc.