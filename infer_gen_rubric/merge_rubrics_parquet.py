#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq


def load_parquet(path: str) -> List[Dict[str, Any]]:
    table = pq.read_table(path)
    return table.to_pylist()


def sanitize_rubric(raw_rubric: Any) -> Dict[str, Any]:
    if raw_rubric is None:
        return {}

    parsed: Any = raw_rubric

    if isinstance(raw_rubric, bytes):
        try:
            parsed = json.loads(raw_rubric.decode("utf-8"))
        except Exception:
            return {}
    elif isinstance(raw_rubric, str):
        raw_rubric = raw_rubric.strip()
        if not raw_rubric:
            return {}
        try:
            parsed = json.loads(raw_rubric)
        except json.JSONDecodeError:
            return {}

    if not isinstance(parsed, dict):
        return {}

    if parsed.get("error"):
        return {}

    return parsed


def _to_hf_image(entry: Any) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        return {"path": None, "bytes": None}

    path = entry.get("path")
    data = entry.get("bytes")

    if isinstance(data, memoryview):
        data = data.tobytes()
        return {"path": path, "bytes": data}
    elif not data:
        return {"path": path}

    # return {"path": path, "bytes": data}


def format_images(images: Any) -> List[Dict[str, Any]]:
    if images is None:
        return []

    if isinstance(images, dict):
        images_iterable = [images]
    elif isinstance(images, list):
        images_iterable = images
    else:
        images_iterable = [images]

    return [_to_hf_image(img) for img in images_iterable]


def merge_records(
    original_records: List[Dict[str, Any]],
    rubric_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rubric_map: Dict[Any, Dict[str, Any]] = {}

    for idx, record in enumerate(rubric_records):
        key = record.get("problem_id", idx)
        rubric_map[key] = sanitize_rubric(record.get("rubric"))

    merged: List[Dict[str, Any]] = []

    for idx, record in enumerate(original_records):
        key = record.get("problem_id", idx)
        rubric = rubric_map.get(key, {})
        answer = record["extra_info"].get("answer", "")
        question_text = record["extra_info"].get("question", "")

        extra_info: Dict[str, Any] = {
            "problem_id": key,
            "answer": answer,
            "question": question_text,
            "question_type": record["extra_info"].get("question_type", "open"),
            "rubric": json.dumps(rubric, ensure_ascii=True) if rubric else None,
        }

        if "split" in record and isinstance(record["split"], str):
            extra_info["split"] = record["split"]
        if "difficulty" in record:
            extra_info["difficulty"] = record["difficulty"]
        merged.append(
            {
                "problem_id": record.get("problem_id", ""),
                "data_source": record.get("data_source", ""),
                "prompt": [
                    {
                        "content": question_text,
                        "role": "user",
                    }
                ],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer,
                },
                "ability": record.get("ability", "math"),
                "images": format_images(record.get("images")),
                "rubric": json.dumps(rubric, ensure_ascii=True),
                "extra_info": extra_info,
            }
        )

    return merged


def save_parquet(records: List[Dict[str, Any]], output_path: str) -> None:
    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge original geometry Parquet with rubric annotations."
    )
    parser.add_argument(
        "--original",
        required=True,
        help="Path to the original Parquet.",
    )
    parser.add_argument(
        "--rubrics",
        required=True,
        help="Path to the rubric Parquet file (no images).",
    )
    parser.add_argument(
        "--output",
        help="Destination Parquet file with merged content.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.output:
        args.output = args.original.replace(".parquet", "_w_rubrics.parquet")

    original_records = load_parquet(args.original)
    rubric_records = load_parquet(args.rubrics)

    merged_records = merge_records(original_records, rubric_records)
    save_parquet(merged_records, args.output)


if __name__ == "__main__":
    main()
