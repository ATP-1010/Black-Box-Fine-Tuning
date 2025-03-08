import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,6"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from tqdm import tqdm, trange
import tinyBenchmarks as tb
import numpy as np
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein
import json
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(1)

# 假设用 "tinyllama" 模型
model1 = "qwen_3b_math"
model2 = "qwen2_1.5b_mmlu" 
model3 = "Qwen/Qwen2-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model3, token='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')

print('Models:', model1, model2, model3)

dataset_name = 'GSM8K'

if dataset_name == 'MMLU':
    tiny_data = load_dataset('tinyBenchmarks/tinyMMLU')['test']
    max_help = 5
elif dataset_name == 'GSM8K':
    tiny_data = load_dataset('tinyBenchmarks/tinyGSM8K', 'main')['test']
    max_help = 256
elif dataset_name == 'ARC-Challenge':
    tiny_data = load_dataset('tinyBenchmarks/tinyAI2_arc', 'ARC-Challenge')['test']
    max_help = 10
elif dataset_name == 'HumanEval':
    tiny_data = load_dataset("openai_humaneval")['test']
    max_help = 512
    save_sample = []
    draft_sample = []
    target_sample = []
    input_sample = []

# 将模型加载到 GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load models to different GPUs
models = [
    AutoModelForCausalLM.from_pretrained(model1, token='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', device_map='auto'),
    AutoModelForCausalLM.from_pretrained(model2, token='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', device_map='auto'),
    AutoModelForCausalLM.from_pretrained(model3, token='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', device_map='auto')
]

# models = [
#     AutoModelForCausalLM.from_pretrained(model1, token='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', device_map='auto'),
#     None,
#     None
# ]

def score_generations(
    hate_classifier_tokenizer, hate_classifier_model, generations, batch_size=32
):
    classifications = []
    for i in trange(0, len(generations), batch_size):
        batch = generations[i: i + batch_size]
        inputs = hate_classifier_tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        ).to(hate_classifier_model.device)
        outputs = hate_classifier_model(**inputs)
        classifications.extend(outputs.logits.argmax(dim=-1).tolist())
    return classifications

def detect_gsm8k_accuracy(output, answer):
    # 匹配第一个以"#### 数字"形式出现的数字
    match_output = re.search(r'#### (\d+)', output)
    match_answer = re.search(r'#### (\d+)', answer)

    # print(match_output, match_answer)
    
    if match_output and match_answer:
        output_number = match_output.group(1)
        answer_number = match_answer.group(1)
        # 返回1如果数字匹配，否则返回0
        return 1 if output_number == answer_number else 0
    return 0  # 如果未找到数字，返回0

def find_numbers(x: str) -> list[str]:
    """Finds all numbers in a string."""
    # Search for number, possibly negative (hyphen), with thousand separators
    # (comma), and with a decimal point (period inbetween digits).
    numbers = re.compile(
        r'-?[\d,]*\.?\d+',
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    ).findall(x)
    return numbers

def find_number(x: str,
                answer_delimiter: str = 'The answer is') -> str:
    """Finds the most relevant number in a string."""
    # If model uses the answer delimiter, then select the first number following
    # that format.
    if answer_delimiter in x:
        answer = x.split(answer_delimiter)[-1]
        numbers = find_numbers(answer)
        if numbers:
            return numbers[0]
    
    # In general, select the last number in the string.
    numbers = find_numbers(x)
    if numbers:
        return numbers[-1]


def maybe_remove_comma(x: str) -> str:
    if x is None:
        return x
    else:
        return x.replace(',', '')
    
def clean_answer(code):
    """
    Borrow from: https://github.com/FSoft-AI4Code/CodeCapybara
    """

    def pad_spaces(s, num=4):
        n = 0
        while n < len(s) and s[n] == " ":
            n += 1
        if n != num:
            s = " " * num + s[n:]
        return s

    # 1. remove everything after "\n\n"
    code = code.split("\n\n")[0]
    # 2. remove everything after the "def "
    code = code.split("def ")[0]
    # 3. pad to four space to avoid `unindent` error
    code = pad_spaces(code, 4)
    return code

def match_option(output_text, options):
    """
    Match the output text to the most relevant option based on Levenshtein distance.

    Parameters:
    - output_text (str): The generated text to be matched.
    - options (dict): A dictionary with "text" (list of options) and "label" (list of labels).

    Returns:
    - str: The label of the most relevant option.
    """
    # Calculate Levenshtein distances between the output and each option
    # print(output_text, options)
    distances = [Levenshtein.distance(output_text, option) for option in options["text"]]

    # Find the index of the minimum distance
    best_match_idx = distances.index(min(distances))
    # print(options["label"][best_match_idx])

    # Return the corresponding label
    return options["label"][best_match_idx]

def gs_score(output_steps, answer_steps):
    correct = 0
    min_len = min(len(answer_steps), len(output_steps))

    for i in range(0, min_len):
        if output_steps[i] == '':
            continue
        try:
            correct += float(maybe_remove_comma(find_number(output_steps[i]))) == float(maybe_remove_comma(find_number(answer_steps[i])))
        except:
            correct += maybe_remove_comma(find_number(output_steps[i])) == maybe_remove_comma(find_number(answer_steps[i]))
    
    return correct/len(answer_steps)

def generate_text_with_custom_ensemble(input_text, max_length=128, stop_token="</s>"):
    # 初始化输入和缓存
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output_ids = input_ids
    past = [None, None, None]  # 仅用于 `a` 模型的 past_key_values
    count, replace = 0, 0

    with torch.no_grad():
        for _ in range(max_length):
            count += 1
            # **b 和 c 不能用 cache**
            output_a = models[0](input_ids, past_key_values=past[0], use_cache=True)
            output_b = models[1](input_ids, past_key_values=past[1], use_cache=True)
            output_c = models[2](input_ids, past_key_values=past[2], use_cache=True)

            # **提取 logits**
            logits_b = output_b.logits[:, -1, :]
            logits_a = output_a.logits[:, -1, :]
            logits_c = output_c.logits[:, -1, :]

            # **更新 past_key_values（仅 a 模型需要）**
            past = [output_a.past_key_values, output_b.past_key_values, output_c.past_key_values]
            # past = [output_a.past_key_values, None, None]

            # **计算 A + B - C 逻辑**
            combined_logits = logits_a[:, :151936].to(device) #+ logits_b.to(device) - logits_c.to(device)
            # combined_logits = logits_a.to(device)# + logits_b.to(device) - logits_c.to(device)
            # if torch.argmax(logits_b, dim=-1).item() != torch.argmax(logits_c, dim=-1).item():
            #     # replace += 1
            if torch.argmax(logits_a, dim=-1).item() != torch.argmax(combined_logits, dim=-1).item():
                replace += 1

            next_token_id = torch.argmax(combined_logits, dim=-1).item()
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

            output_ids = torch.cat([output_ids, next_token_tensor], dim=-1)

            # **终止条件**
            # next_token = tokenizer.decode(next_token_id)
            # print(next_token)
            if next_token_id == 2:
                break

            # **确保 input_ids 只包含最新 token**
            input_ids = next_token_tensor
        # print('Replace Ratio:', replace/count)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# def generate_text_with_custom_ensemble(input_text, max_length=128, stop_token="</s>"):
#     # 初始化输入和缓存
#     input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
#     output_ids = input_ids
#     past = None  # 仅用于 `a` 模型的 past_key_values
#     cache_use = False  # `a` 模型是否使用 past_key_values

#     for _ in range(max_length):
#         with torch.no_grad():
#             # **b 和 c 不能用 cache**
#             output_b = models[1](input_ids)  # b 不使用 past_key_values
#             output_c = models[2](input_ids)  # c 不使用 past_key_values

#             # **a 需要 cache，但要确保正确使用**
#             if past is None or not cache_use:
#                 output_a = models[0](input_ids, use_cache=True)
#             else:
#                 output_a = models[0](input_ids, past_key_values=past, use_cache=True)

#         # **提取 logits**
#         logits_b = output_b.logits[:, -1, :]
#         logits_a = output_a.logits[:, -1, :]
#         logits_c = output_c.logits[:, -1, :]

#         # **更新 past_key_values（仅 a 模型需要）**
#         if hasattr(output_a, "past_key_values"):
#             past = output_a.past_key_values  # 仅 a 更新 past_key_values

#         # **计算 A + B - C 逻辑**
#         combined_logits = logits_a.to(device) + logits_b.to(device) - logits_c.to(device)

#         # **选择最可能的下一个 token**
#         ori_id = torch.argmax(logits_a, dim=-1).item()
#         next_token_id = torch.argmax(combined_logits, dim=-1).item()

#         # **确保 next_token_id 是 Tensor**
#         next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

#         # **cache 逻辑（仅影响 a）**
#         cache_use = (ori_id == next_token_id)

#         # **更新输出**
#         output_ids = torch.cat([output_ids, next_token_tensor], dim=1)

#         # **终止条件**
#         next_token = tokenizer.decode(next_token_id)
#         print(next_token)
#         if next_token == stop_token:
#             break

#         # **确保 input_ids 只包含最新 token**
#         input_ids = output_ids
#         if cache_use:
#             help_id = output_ids[:, -1:].clone().detach().to(device)

#     return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# def generate_text_with_custom_ensemble(input_text, max_length=128, stop_token="</s>", confidence_threshold=1):
#     """
#     生成文本时采用 ensemble 策略：默认 ensemble = a + b - c，
#     但当 a 模型预测的置信度（即预测 token 的 softmax 概率）大于 confidence_threshold 时，
#     直接采用 a 模型的预测结果，从而跳过 b 和 c 模型的计算。

#     参数：
#       input_text: 初始输入文本
#       max_length: 最大生成 token 数
#       stop_token: 停止生成的 token（解码后判断）
#       confidence_threshold: a 模型的置信度阈值，超过该值则直接采用 a 模型的结果
#     """
#     # 编码初始输入（全序列，用于 b 和 c 模型，因为它们不支持缓存）
#     full_input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
#     # a 模型支持缓存时单步输入
#     a_input_ids = full_input_ids.clone()
#     past = None            # 用于存储 a 模型的 past_key_values
#     cache_use = False      # 标识上一步 ensemble 是否与 a 模型输出一致

#     count = 0
#     acc_count = 0

#     with torch.no_grad():
#         for _ in range(max_length):
#             count += 1
#             # ----- 计算 a 模型的输出 -----
#             if past is not None and cache_use:
#                 output_a = models[0](a_input_ids, past_key_values=past, use_cache=True)
#             else:
#                 output_a = models[0](a_input_ids, use_cache=False)
#             logits_a = output_a.logits[:, -1, :]  # shape: (1, vocab_size)
#             # 如果 a 模型支持缓存，则保存它的 past_key_values
#             a_past = output_a.past_key_values if hasattr(output_a, "past_key_values") else None

#             # 计算 a 模型对每个 token 的概率，并获得最大概率及对应的 token
#             # probs_a = torch.softmax(logits_a, dim=-1)
#             # ori_confidence, ori_token = torch.argmax(logits_a, dim=-1)
#             ori_token = torch.argmax(logits_a, dim=-1)
#             # ori_confidence = ori_confidence.item()
#             # print(ori_confidence)
#             ori_id = int(ori_token.item())
#             # next_token_id = ori_id

#             # ----- 判断是否直接使用 a 的预测 -----
#             # if ori_confidence > confidence_threshold:
#             #     # 当 a 模型置信度高时，直接采用 a 的预测，跳过 b 和 c 的计算
#             #     next_token_id = ori_id
#             # 当 a 模型置信度不足时，计算 b 和 c 模型的输出，再进行 ensemble 逻辑：A + B - C
#             output_b = models[1](full_input_ids)
#             output_c = models[2](full_input_ids)
#             logits_b = output_b.logits[:, -1, :]
#             logits_c = output_c.logits[:, -1, :]

#             combined_logits = logits_a + logits_b - logits_c
#             next_token_id = int(torch.argmax(combined_logits, dim=-1).item())

#             # 更新缓存使用逻辑：如果 ensemble 结果和 a 模型自身预测一致，则可继续利用缓存
#             # print(print('A, Final:', ori_id, next_token_id))
#             cache_use = (ori_id == next_token_id)
#             acc_count += (ori_id == next_token_id)

#             # 构造下一个 token 的张量，并更新输入序列（b 和 c 模型使用完整上下文）
#             next_token_tensor = torch.tensor([[next_token_id]], device=device)
#             full_input_ids = torch.cat([full_input_ids, next_token_tensor], dim=1)

#             # 更新 a 模型的输入：如果上一步 ensemble 与 a 模型一致，则只传入最新 token（利用缓存），否则传入完整上下文并重置 past
#             if cache_use:
#                 a_input_ids = next_token_tensor
#                 past = a_past
#             else:
#                 a_input_ids = full_input_ids
#                 past = None

#             # 解码输出并打印
#             next_token = tokenizer.decode(next_token_id)
#             print(next_token, end=" ", flush=True)
#             if next_token_id == 2:
#                 # print(next_token_id)
#                 break
    
#     print("Accept Rate:", acc_count/count)

#     # 返回生成的完整文本（去除特殊 token）
#     return tokenizer.decode(full_input_ids[0], skip_special_tokens=True)

# def generate_text_with_custom_ensemble(input_text, max_length=128, stop_token="</s>", confidence_threshold=1):
#     """
#     生成文本时采用 ensemble 策略：默认 ensemble = a + b - c，
#     但当 a 模型预测的置信度（即预测 token 的 softmax 概率）大于 confidence_threshold 时，
#     直接采用 a 模型的预测结果，从而跳过 b 和 c 模型的计算。

#     参数：
#       input_text: 初始输入文本
#       max_length: 最大生成 token 数
#       stop_token: 停止生成的 token（解码后判断）
#       confidence_threshold: a 模型的置信度阈值，超过该值则直接采用 a 模型的结果
#     """
#     # 编码初始输入（全序列，用于所有模型）
#     full_input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
#     count = 0
#     acc_count = 0

#     for _ in range(max_length):
#         count += 1
#         with torch.no_grad():
#             # ----- 计算 a 模型的输出 -----
#             output_a = models[0](full_input_ids, use_cache=False)
#             logits_a = output_a.logits[:, -1, :]  # shape: (1, vocab_size)

#             # 计算 a 模型预测的 token（以及对应的置信度可以通过 softmax 计算，但这里只取了最大概率对应的 token）
#             ori_token = torch.argmax(logits_a, dim=-1)
#             ori_id = int(ori_token.item())

#             # 如果 a 模型置信度足够高，可直接使用 a 模型的预测（此处逻辑目前注释掉）
#             # if ori_confidence > confidence_threshold:
#             #     next_token_id = ori_id
#             # else:
#             # ensemble：计算 b 和 c 模型的输出，然后使用 ensemble 策略：a + b - c

#             output_b = models[1](full_input_ids, use_cache=False)
#             output_c = models[2](full_input_ids, use_cache=False)
#             logits_b = output_b.logits[:, -1, :]
#             logits_c = output_c.logits[:, -1, :]

#             combined_logits = logits_a# + logits_b - logits_c

#             # b_token = int(torch.argmax(logits_b, dim=-1).item())
#             # c_token = int(torch.argmax(logits_c, dim=-1).item())
#             # print('a:', tokenizer.decode(ori_id))
#             # print('b:', tokenizer.decode(b_token))
#             # print('c:', tokenizer.decode(c_token))
#             next_token_id = int(torch.argmax(combined_logits, dim=-1).item())
#             # print('new:', tokenizer.decode(next_token_id))

#             # # 记录 ensemble 结果与 a 模型预测一致的情况
#             acc_count += (ori_id == next_token_id)

#             # # 更新完整的输入序列（所有模型始终使用完整上下文）
#             next_token_tensor = torch.tensor([[next_token_id]], device=device)
#             full_input_ids = torch.cat([full_input_ids, next_token_tensor], dim=1)

#             # 如果生成的 token 是停止 token（这里假定停止 token 的 ID 为 2），则退出循环

#             # print(tokenizer.decode(next_token_id))
#             if next_token_id == 2:
#                 break

#     print("Accept Rate:", acc_count / count)

#     # 返回生成的完整文本（去除特殊 token）
#     return tokenizer.decode(full_input_ids[0], skip_special_tokens=True)


final_acc = []
for sample_idx in tqdm(range(0, len(tiny_data))):
    if dataset_name == 'HumanEval':
        prompt_text = tiny_data[sample_idx]['prompt']
        ref_text = tiny_data[sample_idx]['canonical_solution']
        input_sample.append(tiny_data[sample_idx])
    else:
        prompt_text = tiny_data[sample_idx]['input_formatted']
            
    # print('Input: ', prompt_text)
    output = generate_text_with_custom_ensemble(prompt_text, max_length=max_help)
    # print(output)
    if len(output) > len(prompt_text) + 1:
        output = output[len(prompt_text)+1: ]
        # print(output)
    else:
        final_acc.append(0)
        continue

    # GSM8K evaluation
    if dataset_name == 'GSM8K':
        final_acc.append(detect_gsm8k_accuracy(output, tiny_data[sample_idx]['answer']))
        # print(final_acc)
    
    #ARC-Challenge
    if dataset_name == 'ARC-Challenge':
        # print(output[0], tiny_data[sample_idx]['answer'])
        best_label = match_option(output.split('\n')[0], tiny_data[sample_idx]['choices'])
        # print('answer:', tiny_data[sample_idx]['answerKey'])
        if tiny_data[sample_idx]['answerKey'] == 'A':
            if best_label == 'A':
                final_acc.append(1)
            else:
                final_acc.append(0)
        elif tiny_data[sample_idx]['answerKey'] == 'B':
            if best_label == 'B':
                final_acc.append(1)
            else:
                final_acc.append(0)
        elif tiny_data[sample_idx]['answerKey'] == 'C':
            if best_label == 'C':
                final_acc.append(1)
            else:
                final_acc.append(0)
        elif tiny_data[sample_idx]['answerKey'] == 'D':
            if best_label == 'D':
                final_acc.append(1)
            else:
                final_acc.append(0)
        elif tiny_data[sample_idx]['answerKey'] == '1':
            if best_label == '1':
                final_acc.append(1)
            else:
                final_acc.append(0)
        elif tiny_data[sample_idx]['answerKey'] == '2':
            if best_label == '2':
                final_acc.append(1)
            else:
                final_acc.append(0)
        elif tiny_data[sample_idx]['answerKey'] == '3':
            if best_label == '3':
                final_acc.append(1)
            else:
                final_acc.append(0)
        elif tiny_data[sample_idx]['answerKey'] == '4':
            if best_label == '4':
                final_acc.append(1)
            else:
                final_acc.append(0)

    # MMLU evaluate
    if dataset_name == 'MMLU':
        # print(output[0], tiny_data[sample_idx]['answer'])
        if tiny_data[sample_idx]['answer'] == 0:
            if output[0] == 'A':
                final_acc.append(1)
            else:
                final_acc.append(0)
        elif tiny_data[sample_idx]['answer'] == 1:
            if output[0] == 'B':
                final_acc.append(1)
            else:
                final_acc.append(0)
        elif tiny_data[sample_idx]['answer'] == 2:
            if output[0] == 'C':
                final_acc.append(1)
            else:
                final_acc.append(0)
        elif tiny_data[sample_idx]['answer'] == 3:
            if output[0] == 'D':
                final_acc.append(1)
            else:
                final_acc.append(0)

    if dataset_name == 'HumanEval':
        save_sample.append({"task_id": tiny_data['task_id'][sample_idx], "completion": clean_answer(output)})

if dataset_name == 'MMLU':
    benchmark = 'mmlu'
    aaa = tb.evaluate(np.array(final_acc), benchmark)
    print('Final::::::::', aaa)
elif dataset_name == 'GSM8K':
    benchmark = 'gsm8k'
    aaa = tb.evaluate(np.array(final_acc), benchmark)
    print('Final::::::::', aaa)
elif dataset_name == 'ARC-Challenge':
    benchmark = 'arc'
    aaa = tb.evaluate(np.array(final_acc), benchmark)
    print('Final::::::::', aaa)
elif dataset_name == 'HumanEval':
    with open("output_humaneval.json", "w", encoding="utf-8") as file:
        for item in save_sample:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.system("evaluate_functional_correctness output_humaneval.json")

# input_text = "Once upon a time"
# generated_text = generate_text_with_custom_ensemble(input_text)
# print(generated_text)

# llama-2-7b + llama7b-mmlu - llama7b: 50.04 47.52 45.36 37.54
# llama13b + llama7b-mmlu - llama7b: 47.64 46.21 45.36 37.54
# llama3-8b + llama3.2-3b - llama3.2-3b  58.43 56.79 49.00

#Final:::::::: {'mmlu': {'irt': 0.4358178631826826, 'pirt': 0.46540163064936835, 'gpirt': 0.4621193751737291}}
#Final:::::::: {'mmlu': {'irt': 0.4879450960891666, 'pirt': 0.47495404592044427, 'gpirt': 0.47639537504117274}}

# llama-2: 47.63 to 54.4

#llama7b-gsm8k: 12.36  7.42

#llama-7b: 'pass@1': 0.12195121951219512
#llama-2-7b: fine-tuned:18.29  ori:16.46(maybe)