import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *
import re

FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    # structs = [dataset.build_prompt(data.iloc[i]) for i in range(lt)]

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'
    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    model = supported_VLM[model_name]() if isinstance(model, str) else model

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)

    for i in tqdm(range(lt)):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i])

        response = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 10 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False, is_boxed_model=False):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    result_file = osp.join(work_dir, f'{model_name}_{dataset_name}.xlsx')

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data(
        model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
        out_file=out_file, verbose=verbose, api_nproc=api_nproc)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        # set the config
        if is_boxed_model:
            data['prediction_in_boxed'] = [extract_predicted_value(data_all[x]) for x in data['index']]
            data['prediction'] = [extract_predicted_output(data_all[x]) for x in data['index']]
        else:
            data['prediction'] = [str(data_all[x]) for x in data['index']]

        data["model_output"] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    if world_size > 1:
        dist.barrier()
    return model



def extract_predicted_output(predict_str: str) -> str:
    """
    Extracts the predicted value from the input string based on the following rules:

    format: <think>...</think>......output here ......
    """
    
    think_match = re.search(r'<think>(.*?)</think>', predict_str, flags=re.DOTALL)
    
    if not think_match:
        return predict_str
    
    end_index = think_match.end()
    if end_index == len(predict_str):
        return predict_str
    
    output_content = predict_str[end_index:]
    return output_content




def extract_predicted_value(predict_str: str) -> str:
    """
    Extracts the predicted value from the input string based on the following rules:

    format: <think>...</think>XXXXX/boxed{answer here}XXXXX

    1) Look for <think> and </think>
        if not found:
            return empty string
    2) Look for /boxed{}
           a) \boxed{\text{...}...} and extract '...' plus any additional text.
           b) \text{\boxed{...}} and extract '\boxed{...}'.
           c) \boxed{...} and extract '...'.
       - If no \boxed{...}:
        return empty string
    
    Examples:
      - "\\boxed{42}" -> "42"
      - "\\boxed{\\text{Light blue}}" -> "Light blue"
      - "<think>xxx</think>\\boxed{42}" -> "42"
      - "<think>}</think>\\boxed{\\text{Light blue}" -> "Light blue"
      - "no boxed content" -> ""
    """
    
    # Step 1: Search for the <think>...</think> block
    think_match = re.search(r'<think>(.*?)</think>', predict_str, flags=re.DOTALL)
    
    if not think_match:
        return ""
    
    # Step 2: Get the end index of the </think> tag
    end_index = think_match.end()

    # # Step 3: Extract the content after the </think> tag
    output_content = predict_str[end_index:]

    # Step 4: Search for \boxed{\text{...}...} with additional text
    box_text_extra_match = re.search(r'\\boxed\{\\text\{(.*?)\}(.*?)\}',output_content, flags=re.DOTALL)
    if box_text_extra_match:
        text_part = box_text_extra_match.group(1)
        text_part = text_part.strip() if text_part is not None else ""
        extra_part = box_text_extra_match.group(2)
        extra_part = extra_part.strip() if extra_part is not None else ""
        # Concatenate with a space if both parts are present
        if extra_part:
            return f"{text_part} {extra_part}"
        return text_part
    
    # Step 2b: Search for \boxed{\text{...}} without additional text
    box_text_match = re.search(r'\\boxed\{\\text\{(.*?)\}\}', output_content, flags=re.DOTALL)
    if box_text_match:
        matched_text = box_text_match.group(1)
        return matched_text.strip() if matched_text is not None else ""
    
    # Step 2c: Search for \text{\boxed{...}} and reconstruct \boxed{...}
    text_box_match = re.search(r'\\text\{\\boxed\{(.*?)\}\}', output_content, flags=re.DOTALL)
    if text_box_match:
        matched_text = text_box_match.group(1)
        matched_text = matched_text.strip() if matched_text is not None else ""
        # Reconstruct \boxed{...}
        return f'\\boxed{{{matched_text}}}'
    
    # Step 2d: Search for \boxed{...} without \text{...}
    box_match = re.search(r'\\boxed\{(.*?)\}', output_content, flags=re.DOTALL)
    if box_match:
        matched_text = box_match.group(1)
        return matched_text.strip() if matched_text is not None else ""

    # If no \boxed{...} found, return empty string
    return ""
