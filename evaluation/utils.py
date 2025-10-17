import json
import random
import time
import zipfile

import argparse
import re
import pyarrow.parquet as pq
import glob
from tqdm import tqdm
from collections import namedtuple
from PIL import Image
import os
import requests

Image.MAX_IMAGE_PIXELS = None
symbols = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]


def cal_acc(anses, gts):
    rws = []
    total = len(anses)
    assert len(anses) == len(gts)
    correct, correct_in_domain, correct_out_domain = 0, 0, 0
    for pid in range(total):
        ans = anses[pid]
        gt = gts[pid]
        try:
            ans = float(ans)
        except:
            pass
        try:
            gt = float(gt)
        except:
            pass
        if ans == gt:
            correct += 1
            rws.append(True)
            continue
        else:
            # print(f"pid: {pid} | gt: {gts[pid]} | ans: {anses[pid]}")
            pass
        rws.append(False)

    return correct / total, rws, total


def option_concat(options):
    refined_options = []
    option_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    if options[0][0] != "A":
        for oid, option in enumerate(options):
            option = f"({option_letters[oid]}): {option}"
            refined_options.append(option)
    else:
        refined_options = options
    concat_options = "\n".join(refined_options)
    final_options = random.choice(
        [f"Choices:\n{concat_options}\n", f"Options:\n{concat_options}\n"]
    )
    return final_options


def extract_one_ans_math(ans):
    ans = ans.strip()
    if ans.endswith("."):
        ans = ans.strip(".")
    if len(ans) == 1:
        return ans
    try:
        fans = float(ans)
        return fans
    except:
        pass
    if "The answer is " in ans:
        ans_extract = ans.split("The answer is ")[-1].rstrip(".")
        try:
            fans = float(ans_extract)
            return fans
        except:
            pass
        ans = ans_extract
        # ans_extract = ans_extract.replace('(', '').replace(')', '')
        # return ans_extract

    match_ABC = re.search(r"\(([A-G])\)", ans)
    if match_ABC:
        res = match_ABC.group(1)
        res = res[:-1] if res.endswith(".") else res
        return res
    match_isABC = re.search(r"is ([A-G])\.", ans)
    if match_isABC:
        res = match_isABC.group(1)
        res = res[:-1] if res.endswith(".") else res
        return res

    match_is_num = re.search(r"is (\d+)", ans)
    if match_is_num:
        return match_is_num.group(1)
    match_equal_num = re.search(r"=(.*?)(\d+)", ans)
    if match_equal_num:
        return match_equal_num.group(2)
    match_str_num = re.search(r"\d+\.\d+", ans)
    if match_str_num:
        return match_str_num.group(0)
    match_num = re.search(r"\d+", ans)
    if match_num:
        return match_num.group(0)
    match_ABClast = re.search(r"\b[A-G]\b\.?$", ans)
    if match_ABClast:
        res = match_ABClast.group(0)
        res = res[:-1] if res.endswith(".") else res
        return res

    match_ans_is = re.search(r"\b(\d+)\b\.?$", ans)
    if match_ans_is:
        res = match_ans_is.group(1)
        res = res[:-1] if res.endswith(".") else res
        return res

    try:
        fans = float(ans.split(" ")[0])
        return fans

    except:
        pass
    return ans


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    samples = []
    with open(path, "r") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def parquet2list(all_files):
    all_data = []
    if isinstance(all_files, list):
        for i, f in enumerate(all_files):
            all_data += parquet2list(f)
        return all_data

    table = pq.read_table(all_files)
    df = table.to_pandas()
    list_of_dicts = df.to_dict("records")
    all_data.extend(list_of_dicts)
    return all_data


def check_image_mode(img_path):
    img = Image.open(img_path)
    if img.mode != "RGB":
        print("converting：", img_path)
        img = img.convert("RGB")
        img.save(img_path)
        print("converted：", img_path)


def save_binary_image(binary_data, img_name):
    with open(img_name, "wb") as file:
        file.write(binary_data)


def download_images(image_url, save_path, timeout=10):
    """
    Downloads images from URLs and saves them locally with specified names.
    """
    try:
        response = requests.get(image_url, stream=True, timeout=timeout)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        else:
            print(f"Failed to download {image_url}: {response.status_code}")
            return False
    except:
        return False


def sort_key(filename):
    match = re.search(r"_(\d+)\.jpg$", filename)
    return int(match.group(1)) if match else float("inf")


def write_jsonl(path, samples, format=False):
    with open(path, "w") as f:
        for sample in samples:
            if format:
                f.write(json.dumps(sample, indent=4, ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(sample) + "\n")


def unzip_one_file(zip_path, file_to_extract, output_dir):
    """
    解压压缩包中的一个文件到指定目录
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        if file_to_extract in zf.namelist():
            zf.extract(file_to_extract, path=output_dir)
            print(f"已解压 {file_to_extract} 到 {output_dir}")
        else:
            print(f"警告：{file_to_extract} 不存在于压缩包 {zip_path} 中。")


def write_json(path, samples, format=False):
    with open(path, "w") as f:
        if format:
            json.dump(samples, f, indent=4, ensure_ascii=False)
        else:
            json.dump(samples, f)


def format_ans(ans):
    return ans.replace("\n", " ").replace("\t", " ").replace("\r", " ")


def remove_duplicates(dict_list):
    seen_ids = set()
    unique_dicts = []
    for item in dict_list:
        if item["id"] not in seen_ids:
            unique_dicts.append(item)
            seen_ids.add(item["id"])

    return unique_dicts


word_num_dic = {
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
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def ans_match(gemini_short_ans, gt):
    gemini_short_ans = str(gemini_short_ans)
    reformed_ans = extract_one_ans_math(gemini_short_ans)
    try:
        if (
            float(reformed_ans) == float(gt)
            or f"{float(reformed_ans):.2f}" == f"{float(gt):.2f}"
        ):
            return True
    except:
        pass
    reformed_gt = str(gt).replace("(", "").replace(")", "")
    if (
        reformed_ans == gt
        or gemini_short_ans == gt
        or reformed_ans == reformed_gt
        or gemini_short_ans == reformed_gt
    ):
        return True
    if isinstance(reformed_ans, str):
        reformed_ans = reformed_ans.lower()
        for word in word_num_dic.keys():
            if word in reformed_ans:
                return ans_match(word_num_dic[word], reformed_gt)

    # print(f"gemini_short_ans: {gemini_short_ans} gt: {gt}\n"
    #       f"reformed_ans: {reformed_ans} reformed_gt: {reformed_gt}")
    return False


def get_time():
    time_now = time.time()
    return str(time_now).split(".")[0]


def write_res(output_file_path, anses, details):
    with open(output_file_path, "w") as aw:
        aw.writelines("\n".join(anses))

    output_save_data = output_file_path.replace(".txt", ".json")
    with open(output_save_data, "w") as aw:
        json.dump(details, aw, indent=4, ensure_ascii=False)


def split_image(image_path):
    # Load the image
    image = Image.open(image_path)
    width, height = image.size

    # Calculate the aspect ratio
    aspect_ratio = max(width / height, height / width)

    # Initialize list to store subimage paths
    subimages = []
    target_ratio = 1
    # Determine the long side and number of parts to split
    if aspect_ratio > target_ratio:
        if width < height:
            long_side = height
            short_side = width
        else:
            long_side = width
            short_side = height

        # Calculate the number of parts
        n = 1
        while (long_side / (n + 1) / short_side) > target_ratio:
            n += 1
        # Calculate size of each part
        part_size = long_side // n
        # Split and save images
        for i in range(n):
            if width < height:
                top = i * part_size
                bottom = top + part_size if i < n - 1 else height
                box = (0, top, width, bottom)
            else:
                left = i * part_size
                right = left + part_size if i < n - 1 else width
                box = (left, 0, right, height)
            subimage = image.crop(box)
            subimage_path = f"{os.path.splitext(image_path)[0]}_subimg{i + 1}.png"
            subimage.save(subimage_path)
            subimages.append(subimage_path)

        return subimages

    else:
        return [image_path]


def check_images_exist(image_dir, check_count=10):
    """
    Check if images already exist in the directory.

    Args:
        image_dir: Directory to check for images
        check_count: Number of files to check to verify they are images

    Returns:
        bool: True if images exist, False otherwise
    """
    if not os.path.exists(image_dir):
        return False

    files = os.listdir(image_dir)
    if not files:
        return False

    # Check if files are actually images by looking at extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    image_files = [
        f for f in files if os.path.splitext(f.lower())[1] in image_extensions
    ]

    if len(image_files) < min(check_count, len(files)):
        return False

    # Check first few files to ensure they are valid image files
    for i, filename in enumerate(image_files[:check_count]):
        filepath = os.path.join(image_dir, filename)
        if not os.path.isfile(filepath):
            return False
        # Basic check - file should have some size
        if (
            os.path.getsize(filepath) < 100
        ):  # Less than 100 bytes is likely not a valid image
            return False

    return True
