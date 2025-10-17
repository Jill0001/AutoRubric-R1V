import os
from tqdm import tqdm
import glob
import pyarrow.parquet as pq
import random
from utils import read_json, parquet2list, save_binary_image, check_images_exist
from utils_eval import symbols, option_refine

open_hint = "Hint: Please answer the question and provide the final answer, e.g., 1.23, 1.34, 1.45, at the end.\n"
mc_hint = "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n"

eval_data_root = (
    "YOUR_EVAL_DATA_ROOT_PATH/"  # Please set your evaluation data root path here
)


def load_data_mathvision(root_dir=eval_data_root + "mathvision"):

    to_be_removed = [
        "<image1>",
        "<image2>",
        "<image3>",
        "<image4>",
        "<image5>",
        "<image6>",
        "<image7>",
        "<image8>",
    ]

    samples = parquet2list(
        os.path.join(root_dir, "test-00000-of-00001-3532b8d3f1b4047a.parquet")
    )

    for sample in samples:
        sample["pid"] = sample["id"]
        sample["image"] = os.path.join(root_dir, sample["image"])
        if not os.path.exists(sample["image"]):
            os.sys("cd {root_dir} && unzip -q images.zip")
        question = sample["question"]
        for _ in to_be_removed:
            question = question.replace(_, "")
        question = question.replace("\n", "")
        sample["query"] = question
        sample["gt"] = sample["answer"]
        sample["choices"] = sample["options"]
        if len(sample["choices"]) > 0:
            sample["eval_type"] = "multiple-choice"
            reformed_options = option_refine(sample["choices"])
            sample["query"] = f"{mc_hint}{question}{reformed_options}"
        else:
            sample["eval_type"] = "open-ended"
            sample["query"] = f"{open_hint}{question}"

    return samples


def load_data_mathvista(root_dir=eval_data_root + "mathvista"):
    samples = parquet2list(
        os.path.join(root_dir, "testmini-00000-of-00001-725687bf7a18d64b.parquet")
    )
    image_root = os.path.join(root_dir, "images")
    if not os.path.exists(image_root):
        os.sys(f"cd {root_dir} && unzip -q images.zip")
    new_samples = []
    for sample in samples:
        new_sample = {}
        new_sample["pid"] = sample["pid"]
        new_sample["image"] = os.path.join(image_root, f"{sample['pid']}.jpg")
        new_sample["query"] = sample["query"]
        new_sample["gt"] = sample["answer"]
        if sample["choices"] is not None:
            symbol_choice_map = {
                sample["choices"][i]: symbol
                for i, symbol in enumerate(symbols[: len(sample["choices"])])
            }
            new_sample["choices"] = sample["choices"]
            new_sample["gt"] = symbol_choice_map[sample["answer"]]
            new_sample["eval_type"] = "multiple-choice"
        else:
            new_sample["eval_type"] = "open-ended"

        new_samples.append(new_sample)

    return new_samples


def load_data_mathverse(root_dir=eval_data_root + "mathverse"):
    samples = read_json(f"{root_dir}/testmini.json")
    image_root = f"{root_dir}/images"

    for sample in samples:
        sample["pid"] = sample["sample_index"]
        sample["image"] = os.path.join(image_root, sample["image"])
        if not os.path.exists(sample["image"]):
            os.sys(f"cd {root_dir} && unzip -q images.zip")
        sample["query"] = sample["query_cot"]
        sample["gt"] = sample["answer"]
        if sample["question_type"] == "multi-choice":
            sample["eval_type"] = "multiple-choice"
            if "Choices:\n" in sample["question_for_eval"]:
                choices_str = sample["question_for_eval"].split("Choices:\n")[-1]
                choices_list = [choice[2:] for choice in choices_str.split("\n")]
                sample["choices"] = choices_list
            elif "\nChoice:\n" in sample["question_for_eval"]:
                choices_str = sample["question_for_eval"].split("\nChoice:\n")[-1]
                choices_list = [choice[2:] for choice in choices_str.split("\n")]
                sample["choices"] = choices_list
            else:
                if sample["pid"] in ["964", "965"]:
                    sample["choices"] = ["8", "6", "4", "2"]
                elif sample["pid"] in ["1693", "1694", "1695"]:
                    sample["choices"] = [
                        "090^\\circ T",
                        "180^\\circ T",
                        "049^\\circ T",
                        "041^\\circ T",
                    ]
                elif sample["pid"] in ["1698"]:
                    sample["choices"] = [
                        "053 ^\\circ T",
                        "082 ^\\circ T",
                        "045 ^\\circ T",
                        "037 ^\\circ T",
                    ]
            sample["choices"] = [
                choice.lstrip(" ").rstrip(" ") for choice in sample["choices"]
            ]

            if sample["gt"] in sample["choices"]:
                answer_index = sample["choices"].index(sample["gt"])
                sample["gt"] = symbols[answer_index]

            if "(" in sample["gt"] and ")" in sample["gt"]:
                sample["gt"] = sample["gt"].replace("(", "").replace(")", "")
            if sample["gt"] not in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
                sample["eval_type"] = "open-ended"
        else:
            sample["eval_type"] = "open-ended"
            sample["choices"] = None
    return samples


def load_data_wemath(root_dir=eval_data_root + "wemath"):
    samples = parquet2list(
        os.path.join(root_dir, "testmini-00000-of-00001-adbfe22dcd3558e6.parquet")
    )
    image_root = os.path.join(root_dir, "images")
    os.makedirs(image_root, exist_ok=True)

    # Check if images already exist
    if check_images_exist(image_root):
        print(f"Images already exist in {image_root}, skipping image saving.")
        skip_image_saving = True
    else:
        skip_image_saving = False

    new_samples = []
    pid_choices_dic = {
        59: ["①", "②", "③", "Cannot be determined", "No correct answer"],
        817: ["①; ③", "②; ③", "②; ④", "①; ④", "No correct answer"],
        1018: ["4", "3", "2", "1", "No correct answer"],
        1019: ["4", "3", "2", "1", "No correct answer"],
    }
    for sample in tqdm(samples):

        new_sample_dic = {}
        img_name = os.path.join(image_root, f"{sample['question number']}.jpg")
        if not skip_image_saving:
            image_binary = sample["image_path"]["bytes"]
            save_binary_image(image_binary, img_name)
        new_sample_dic["image"] = img_name
        new_sample_dic["pid"] = sample["question number"]

        ori_options = sample["option"].split(";")
        choices_list = []
        for opt in ori_options:
            if ". " not in opt:
                choices_list = pid_choices_dic[new_sample_dic["pid"]]
            else:
                opt = opt.split(". ")[1]
                choices_list.append(opt)
        new_sample_dic["choices"] = choices_list
        new_sample_dic["gt"] = sample["answer"]
        new_sample_dic["eval_type"] = "multiple-choice"
        choices_text = option_refine(new_sample_dic["choices"])
        new_sample_dic["query"] = f"{sample["question"]}{choices_text}"
        new_samples.append(new_sample_dic)
    return new_samples


def load_data_mmmu(root_dir=eval_data_root + "mmmu"):
    all_data = []
    all_files = glob.glob(os.path.join(root_dir, "MMMU/*/val*.parquet"))
    for f in all_files:
        table = pq.read_table(f)
        df = table.to_pandas()
        list_of_dicts = df.to_dict("records")
        all_data.extend(list_of_dicts)

    # Check if images already exist
    image_dir = os.path.join(root_dir, "images")
    if check_images_exist(image_dir):
        print(f"Images already exist in {image_dir}, skipping image saving.")
        skip_image_saving = True
    else:
        skip_image_saving = False

    new_samples = []
    for exp in tqdm(all_data):
        qid = exp["id"]
        question, options, answers = (
            exp["question"],
            eval(exp["options"]),
            exp["answer"],
        )
        concated_options = option_refine(options)
        images_path = []
        find_image = f"{question} {concated_options}"
        for i in range(7):
            if exp[f"image_{i + 1}"] is not None and f"<image {i + 1}>" in find_image:
                img_name = f"images/{qid}_{i}.png"
                img_path = os.path.join(root_dir, img_name)
                if not skip_image_saving:
                    binary_data = exp[f"image_{i + 1}"]["bytes"]
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    with open(img_path, "wb") as file:
                        file.write(binary_data)
                images_path.append(img_path)

        for i in range(7):
            question = question.replace(f"<image {i + 1}>", "<image>")
            concated_options = concated_options.replace(f"<image {i + 1}>", "<image>")
        if exp["question_type"] == "multiple-choice":
            query = f"{mc_hint}{question}{concated_options}"
        else:
            query = f"{open_hint}{question}"
        if exp["question_type"] == "open":
            exp["question_type"] = "open-ended"

        new_sample_dic = {}
        new_sample_dic["image"] = images_path
        new_sample_dic["pid"] = qid
        new_sample_dic["choices"] = options
        new_sample_dic["gt"] = answers
        new_sample_dic["eval_type"] = exp["question_type"]
        new_sample_dic["query"] = query
        new_samples.append(new_sample_dic)

    print(f"Prepared {len(new_samples)} samples for MMMU dataset.")
    return new_samples


def load_data_mmmu_pro(root_dir=eval_data_root + "mmmu_pro"):
    parquet_files = [
        os.path.join(root_dir, f"test-{i:05d}-of-00004.parquet") for i in range(4)
    ]
    all_samples = []
    for f in parquet_files:
        samples = parquet2list(f)
        all_samples.extend(samples)

    # Check if images already exist
    image_dir = os.path.join(root_dir, "images")
    if check_images_exist(image_dir):
        print(f"Images already exist in {image_dir}, skipping image saving.")
        skip_image_saving = True
    else:
        skip_image_saving = False

    new_samples = []
    for sample in tqdm(all_samples):
        new_sample_dic = {}
        image_bytes = sample["image"]
        image_dir = os.path.join(root_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        image_name = f"{image_dir}/{sample['id']}.png"
        if not skip_image_saving:
            save_binary_image(image_bytes["bytes"], image_name)
        new_sample_dic["image"] = image_name
        new_sample_dic["pid"] = sample["id"]
        new_sample_dic["choices"] = eval(sample["options"])
        options_text = option_refine(new_sample_dic["choices"])
        new_sample_dic["gt"] = sample["answer"]
        new_sample_dic["eval_type"] = "multiple-choice"
        new_sample_dic["query"] = (
            f"{mc_hint}Please answer the question in the image.{options_text}"
        )
        new_samples.append(new_sample_dic)
    return new_samples


if __name__ == "__main__":
    samples = load_data_mathvista()
    for sample in random.sample(samples, 1):
        print(sample)
