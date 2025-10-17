#!/usr/bin/env python3
"""
测试 rl_dataset.py 修改效果的脚本
用法: python test_dataset.py /path/to/your/data.parquet
"""

import os
import sys
from omegaconf import OmegaConf

# 添加 verl 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verl.utils import hf_tokenizer
from verl.utils.dataset.rl_dataset import RLHFDataset


def main():
    # 获取数据文件路径
    if len(sys.argv) < 2:
        print("用法: python test_dataset.py /path/to/your/data.parquet")
        sys.exit(1)

    data_file = sys.argv[1]
    if not os.path.exists(os.path.expanduser(data_file)):
        print(f"错误: 文件不存在 - {data_file}")
        sys.exit(1)

    print(f"使用数据文件: {data_file}")

    # 配置（模拟训练时的配置）
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 2048,
            "filter_overlong_prompts": False,  # 关闭过滤以加快测试速度
            "return_raw_chat": True,
            "cache_dir": "~/.cache/verl/rlhf",
        }
    )

    # 检测是否是多模态数据集（包含 geo3k 等关键字）
    is_multimodal = (
        "geo3k" in data_file.lower()
        or "image" in data_file.lower()
        or "vision" in data_file.lower()
    )

    if is_multimodal:
        # 使用多模态模型
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        print(f"检测到多模态数据，使用模型: {model_name}")
        from verl.utils import hf_processor

        tokenizer = hf_tokenizer(model_name, trust_remote_code=True)
        processor = hf_processor(model_name, trust_remote_code=True, use_fast=True)
    # else:
    #     # 使用纯文本模型
    #     model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    #     print(f"使用文本模型: {model_name}")
    #     tokenizer = hf_tokenizer(model_name, trust_remote_code=True)
    #     processor = None

    # 创建数据集
    print("创建数据集...")
    dataset = RLHFDataset(
        data_files=data_file,
        tokenizer=tokenizer,
        processor=processor if is_multimodal else None,
        config=config,
    )

    print(f"数据集大小: {len(dataset)}")
    print("=" * 60)

    # 测试前3个样本
    num_samples = min(3, len(dataset))
    for i in range(num_samples):
        print(f"\n样本 {i}:")
        print("-" * 40)

        item = dataset[i]

        # 显示原始 messages
        if "raw_prompt" in item:
            print("\n【原始 messages】")
            for msg in item["raw_prompt"]:
                role = msg.get("role", "unknown")
                content = str(msg.get("content", ""))
                # 处理可能的列表内容（多模态）
                if isinstance(content, list):
                    content = str(content[0]) if content else ""
                # 只显示前150个字符
                display_content = (
                    content[:150] + "..." if len(content) > 150 else content
                )
                print(f"  {role}: {display_content}")

        # 解码并显示处理后的 prompt
        if "input_ids" in item:
            decoded = tokenizer.decode(item["input_ids"], skip_special_tokens=False)
            print(f"\n【处理后的 prompt】(前500字符)")
            print(decoded[:500])

            # 检查你在 rl_dataset.py 中添加的修改是否生效
            print("\n【检查修改是否生效】")
            modifications = {
                "[INSTRUCTION]": "[INSTRUCTION]" in decoded,
                "helpful and harmless": "helpful and harmless" in decoded,
                "step by step": "step by step" in decoded,
                "Difficulty level": "Difficulty level" in decoded,
                "calculation process": "calculation process" in decoded,
                "working code": "working code" in decoded,
            }

            for key, found in modifications.items():
                status = "✓" if found else "✗"
                print(f"  {status} {key}")

        # 显示数据中的其他字段
        print("\n【其他字段】")
        for key in [
            "problem_id",
            "problem_type",
            "difficulty",
            "require_cot",
            "answer",
        ]:
            if key in item:
                value = str(item[key])
                if len(value) > 100:
                    value = value[:100] + "..."
                print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("\n提示: 修改 verl/utils/dataset/rl_dataset.py 的 __getitem__ 方法后，")
    print("重新运行此脚本即可看到新效果，无需重启训练程序。")


if __name__ == "__main__":
    main()
