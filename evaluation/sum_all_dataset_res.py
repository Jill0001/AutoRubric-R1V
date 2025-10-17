import os
import json


def summarize_final_acc(
    base_dir: str, output_filename: str = "eval_summary.jsonl"
) -> dict:

    eval_results_dir = os.path.join(base_dir, "eval_results")
    summary_list = []
    average_acc = 0.0
    for folder_name in os.listdir(eval_results_dir):
        folder_path = os.path.join(eval_results_dir, folder_name)
        if os.path.isdir(folder_path):
            final_acc_path = os.path.join(folder_path, "final_acc.json")
            if os.path.isfile(final_acc_path):
                try:
                    with open(final_acc_path, "r") as f:
                        acc_data = json.load(f)

                    # 提取数据集名称前缀，例如 mathverse_qwen25... -> mathverse
                    dataset_name = folder_name

                    entry = {
                        "dataset": dataset_name,
                        "acc": acc_data.get("acc"),
                        "correct": acc_data.get("correct"),
                        "total": acc_data.get("total"),
                        "eval_type_stats": acc_data.get("eval_type_stats", {}),
                    }
                    average_acc += float(entry["acc"].strip("%"))
                    summary_list.append(entry)
                except Exception as e:
                    print(f"读取 {final_acc_path} 时出错: {e}")

    output_path = os.path.join(eval_results_dir, output_filename)
    out_f = open(output_path, "w")
    summary_list = sorted(summary_list, key=lambda x: x["dataset"])
    for entry in summary_list:
        out_f.write(json.dumps(entry) + "\n")

    print(f"Summarized into {output_path}")
    avg_across_dataset = average_acc / len(summary_list)
    print(f"Average accuracy across all datasets: {avg_across_dataset:.2f}%")
    avg_entry = {"dataset": "average", "acc": f"{avg_across_dataset:.2f}%"}
    out_f.write(json.dumps(avg_entry) + "\n")


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--save_dir")
args = parser.parse_args()
model_path = args.save_dir
summarize_final_acc(model_path)
