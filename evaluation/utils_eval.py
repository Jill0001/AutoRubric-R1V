import argparse
import json
import os
import sys
import editdistance

from PIL import Image

from utils import read_json, write_json, read_jsonl, write_jsonl


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def anls_metric(target: str, prediction: str, theta: float = 0.5):
    """Calculates ANLS for DocVQA.

    There does not seem to be an official evaluation script.
    Public implementation on which this implementation is based:
    https://github.com/herobd/layoutlmv2/blob/main/eval_docvqa.py#L92

    Original paper (see Eq 1): https://arxiv.org/pdf/1907.00490.pdf

    Args:
        target: Target string.
        prediction: Predicted string.
        theta: Filter threshold set to 0.5 for DocVQA.

    Returns:
        ANLS score.
    """

    edit_distance = editdistance.eval(target, prediction)
    normalized_ld = edit_distance / max(len(target), len(prediction))
    return 1.0 - normalized_ld if normalized_ld < theta else 0.0


def metric_calculate(targets, prediction):
    if len(targets) == 0:
        return 1 if prediction in ["", "none", "NA", None, []] else 0
    if len(prediction) == 0:
        return 0

    p = prediction.lower()
    score = max(anls_metric((t.lower()), p) for t in targets)
    return score


def calculate_anls(gt, pred):
    if len(gt) == 0:
        return 1 if pred in ["", "none", "NA", "unanswerable", None, []] else 0
    if len(pred) == 0:
        return 0

    # if answer_type == 'not-answerable':
    #     return 1 if pred in ['', 'none', 'NA', None, []] else 0

    # if pred == 'none' and answer_type != 'not-answerable':
    #     return 0
    answers_similarity = [
        1 - editdistance.eval(gt_elm, pred) / max(len(gt_elm), len(pred))
        for gt_elm in gt
    ]
    max_similarity = max(answers_similarity)
    anls = max_similarity if max_similarity >= 0.5 else 0
    return anls


symbols = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]


def format_acc(correct, total):
    return f"{(correct / total) * 100:.2f}%"


def split_shard(all_data, args):
    shard_size = len(all_data) // args.num_shards + 1
    all_data = all_data[args.shard * shard_size : (args.shard + 1) * shard_size]
    return all_data


def option_refine(options):
    if options is None:
        return ""
    refined_options = []
    for oid, option in enumerate(options):
        option = f"{symbols[oid]}. {option}\n"
        refined_options.append(option)
    concated_option = "".join(refined_options)
    concated_option_final = f"\nChoices:\n{concated_option}"
    return concated_option_final


def get_instruction(setting, ques_type):
    if ques_type == "multiple-choice":
        if setting == "cot":
            instruction = "First think step by step. Then answer with the letter of the correct option."
        elif setting == "direct":
            instruction = (
                "Answer with the option's letter from the given choices directly."
            )
        elif setting == "none":
            instruction = ""
    else:
        if setting == "cot":
            instruction = (
                "First think step by step. Then answer with a single word or phrase."
            )
        elif setting == "direct":
            instruction = "Answer the question using a single word or phrase."
        elif setting == "none":
            instruction = ""
    return instruction


def retain_n_images(text, n):
    count = 0

    def replacer(match):
        nonlocal count
        count += 1
        if count <= n:
            return match.group(0)
        else:
            return ""

    pattern = r"<image>"
    result = re.sub(pattern, replacer, text)
    return result


def cut_img(image):
    ori_w, ori_h = image.size

    if ori_w <= 672 and ori_h <= 672:
        sub_images = [image]
        return sub_images
    re_w, re_h = ori_w // 2, ori_h // 2

    sub_images = [image]
    for i in range(2):
        for j in range(2):
            left = j * re_w
            upper = i * re_h
            right = left + re_w
            lower = upper + re_h

            sub_image = image.crop((left, upper, right, lower))
            sub_images.append(sub_image)
    return sub_images


def change_token_in_json():
    config_json_path = os.path.join(args.checkpoint, "config.json")
    with open(config_json_path, "r") as f:
        config = json.load(f)
    config["image_token_index"] = 128200
    with open(config_json_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Write to {config_json_path} done")


from collections import defaultdict


def group_acc(args):
    evaL_setting_name = args.setting
    eval_bench_name = args.dataset

    input_files_details = [
        f
        for f in os.listdir(args.checkpoint)
        if f"{evaL_setting_name}_{eval_bench_name}_shard_details.jsonl" in f
    ]
    if len(input_files_details) == 0:
        print("No splitted files found")

    all_res_details = []
    for file in input_files_details:
        all_res_details.extend(read_jsonl(os.path.join(args.checkpoint, file)))

    remove_cmds = [
        f"rm {os.path.join(args.checkpoint, file)}" for file in input_files_details
    ]
    correct, total, multi_img_correct, multi_img_total, correct_anls = 0, 0, 0, 0, 0
    image_type_acc_dic = {}
    stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for res_dic in all_res_details:
        correct += res_dic["correct"]
        total += 1

        if "correct_anls" in res_dic:
            if res_dic["correct_anls"] != None:
                correct_anls += res_dic["correct_anls"]
        else:
            correct_anls = 0

        if res_dic["multi_img"]:
            stats["multi_img"]["total"] += 1
            if res_dic["correct"]:
                stats["multi_img"]["correct"] += 1
        else:
            stats["single_img"]["total"] += 1
            if res_dic["correct"]:
                stats["single_img"]["correct"] += 1

        if "image_type" in res_dic and res_dic["image_type"] is not None:
            t = res_dic["image_type"]
            stats[t]["total"] += 1
            if res_dic["correct"]:
                stats[t]["correct"] += 1

    type_acc_dic = {}
    for t, counts in stats.items():
        accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        type_acc_dic[f"image type {t} acc"] = f"{accuracy:.2%}"
        type_acc_dic[f"image type {t} total"] = counts["total"]

    acc = round((correct / total) * 100, 2)
    acc_anls = round((correct_anls / total) * 100, 2)
    merged_res_dic = {
        "Acc": acc,
        "Total": total,
        "Correct": correct,
        "Acc_anls": acc_anls,
    }
    merged_res_dic_multi = {}

    merged_res_dic.update(merged_res_dic_multi)
    merged_res_dic.update(type_acc_dic)

    to_be_print = ""
    for key, value in merged_res_dic.items():
        to_be_print += f"{key}:{value} |"
    print(args.dataset, to_be_print)

    write_jsonl_path = os.path.join(
        args.checkpoint, f"{args.dataset}_{args.setting}_acc.json"
    )
    print(f"write to {write_jsonl_path}")
    write_json(write_jsonl_path, merged_res_dic)
    write_jsonl(
        write_jsonl_path.replace("_acc.json", "_details.jsonl"), all_res_details
    )


"""Response Parsing and Evaluation for various models"""
from typing import Dict

import re
import random

random.seed(42)
import numpy as np


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response or f"{choice}." in response:
                candidates.append(choice)
    if len(candidates) == 0:
        for choice in all_choices:
            if f"：{choice}" in response or f"Answer:{choice}" in response:
                candidates.append(choice)
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}: " in response or f"{choice}. " in response:
                candidates.append(choice)
    if len(candidates) == 0:
        for index, ans in index2ans.items():
            if (
                ans.lower() in response.lower()
                or response.lstrip(" ").rstrip(" ").lower() in ans.lower()
            ):
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        pred_index = "None"
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]

    return pred_index


def check_is_number(string):
    """
    Check if the given string a number.
    """
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        return False


def normalize_str(string):
    """
    Normalize the str to lower case and make them float numbers if possible.
    """
    # check if characters in the string

    # if number, numerize it.
    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(",", "")
        string = float(string)
        # leave 2 decimal
        string = round(string, 2)
        return [string]
    else:  # it's likely to be a string
        # lower it
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]  # avoid trivial matches
        return [string]


def extract_numbers(string):
    """
    Exact all forms of numbers from a string with regex.
    """
    # Pattern for numbers with commas
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    # Pattern for scientific notation
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    # Pattern for simple numbers without commas
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"

    # Extract numbers with commas
    numbers_with_commas = re.findall(pattern_commas, string)
    # Extract numbers in scientific notation
    numbers_scientific = re.findall(pattern_scientific, string)
    # Extract simple numbers without commas
    numbers_simple = re.findall(pattern_simple, string)

    # Combine all extracted numbers
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers


def parse_open_response(response):
    """
    Parse the prediction from the generated response.
    Return a list of predicted strings or numbers.
    """

    # content = content.strip("\n").strip(".").strip(" ")
    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)
        indicators_of_keys = [
            "answer: ",
            "Answer: ",
            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
        ]
        key_responses = []
        for index, resp in enumerate(sub_responses):
            # if last one, accept it's an equation (the entire response can be just one sentence with equation)
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(["="])
            shortest_key_response = None  # the shortest response that may contain the answer (tail part of the response)
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(
                            shortest_key_response
                        ):
                            shortest_key_response = resp.split(indicator)[-1].strip()
                    # key_responses.append(resp.split(indicator)[1].strip())

            if shortest_key_response:
                # and it's not trivial
                if shortest_key_response.strip() not in [
                    ":",
                    ",",
                    ".",
                    "!",
                    "?",
                    ";",
                    ":",
                    "'",
                ]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:  # did not found any
            return [response]
        return key_responses

    # pdb.set_trace()
    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()  # keep the original string response
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    pred_list = tmp_pred_list

    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list


# ----------- Evaluation -------------


def eval_multi_choice(gold_i, pred_i):
    """
    Evaluate a multiple choice instance.
    """
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return 1 if correct else 0


def eval_open(gold_i, pred_i):
    """
    Evaluate an open question instance
    """
    correct = False
    word_to_number = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }
    if isinstance(gold_i, list):
        # use float to avoid trivial matches
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)

    for pred in pred_i:  # pred is already normalized in parse response phase
        if pred in word_to_number:
            pred = str(word_to_number[pred])

        if isinstance(pred, str):  # if it's a string, then find if ans in the pred_i
            for norm_ans in norm_answers:
                # Check if norm_ans is a word representing a number
                if (
                    isinstance(norm_ans, str)
                    and norm_ans.lower().strip(".") in word_to_number
                ):
                    norm_ans = str(word_to_number[norm_ans.lower().strip(".")])
                # only see if the string answer in the string pred
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
                if isinstance(norm_ans, str) and norm_ans.strip(".") in pred:
                    if not correct:
                        correct = True
                    break
            # for norm_ans in norm_answers:
            #     if isinstance(norm_ans, str):
            #         if str(pred) in norm_ans:
            #             if not correct:
            #                 correct = True
            #             break

        else:  # it's a float number
            if pred in norm_answers:
                if not correct:
                    correct = True
                break

    return 1 if correct else 0


def eval_number(gold_i, pred_i):
    """
    Evaluate an open question instance
    """
    for pred in pred_i:  # pred is already normalized in parse response phase
        try:
            pred = float(pred)
        except:
            continue
        if float(gold_i) == float(pred):
            return 1
    return 0


from rouge import Rouge

rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])


def eval_rouge(gold_i, pred_i):
    scores = rouge.get_scores(pred_i, gold_i, avg=True)
    return scores["rouge-l"]["f"]


import pprint


#
def merge_all_bench_results(all_results_dir):
    acc_json_files = []
    bench_reses = {}
    for filename in os.listdir(all_results_dir):
        if "_acc.json" in filename and not filename[0].isdigit():
            acc_json_files.append(filename)
    for res_file in acc_json_files:
        cur_bench_name = res_file.replace("_acc.json", "")
        bench_reses[cur_bench_name] = read_json(os.path.join(all_results_dir, res_file))
    bench_reses = dict(sorted(bench_reses.items()))
    merged_file = os.path.join(all_results_dir, "merged_res.json")
    pprint.pprint(bench_reses)
    write_json(merged_file, bench_reses, format=True)
    return acc_json_files


# ----------- Batch Evaluation -------------
def evaluate(samples):
    """
    Batch evaluation for multiple choice and open questions.
    """
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        if sample["question_type"] == "multiple-choice":
            correct = eval_multi_choice(gold_i, pred_i)
        else:  # open question
            correct = eval_open(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}


# ----------- Calculate Accuracy -------------
def calculate_ins_level_acc(results: Dict):
    """Calculate the instruction level accuracy for given Subject results"""
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
    if ins_num == 0:
        return 0
    return acc / ins_num


def eval_one_sample(response, answers, eval_type, options=None):
    if eval_type == "multiple-choice":
        chosen = parse_multi_choice_response(
            response,
            symbols[: len(options)],
            {s: o for s, o in zip(symbols[: len(options)], options)},
        )
        correct = eval_multi_choice(answers, chosen)
    elif eval_type == "open-ended":
        if not isinstance(answers, list):
            answers = [answers]
        chosen = parse_open_response(response)
        # chosen.append(response)
        correct = eval_open(answers, chosen)
    elif eval_type == "number":

        chosen = parse_open_response(response)
        # chosen.append(response)
        correct = eval_number(answers, chosen)
    elif eval_type == "anls":
        formated_response = response.split("Answer: ")[-1].lower()
        chosen = formated_response
        answers = [answer.lower() for answer in answers]

        correct = calculate_anls(gt=answers, pred=formated_response)
    elif eval_type == "captioning":
        chosen = response

        correct = eval_rouge([answers[0].lower()], [response.lower()])
    else:
        raise ValueError(f"Invalid question type: {eval_type}")

    return correct, chosen
