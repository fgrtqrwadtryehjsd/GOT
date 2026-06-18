"""
测试模块6：数据集准备 data/prepare_data

覆盖：
- _clutrr_builtin_samples
- load_processed_dataset（本地缓存路径）
- validate_dataset
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.prepare_data import (
    _clutrr_builtin_samples,
    load_processed_dataset,
    validate_dataset,
)


class TestClutrBuiltin:
    def test_returns_list(self):
        samples = _clutrr_builtin_samples(5)
        assert isinstance(samples, list)
        assert len(samples) == 5

    def test_required_fields(self):
        samples = _clutrr_builtin_samples(3)
        for s in samples:
            assert "question" in s
            assert "answer" in s
            assert "context" in s

    def test_unique_ids(self):
        samples = _clutrr_builtin_samples(10)
        ids = [s["id"] for s in samples]
        assert len(ids) == len(set(ids))

    def test_zero_samples(self):
        samples = _clutrr_builtin_samples(0)
        assert samples == []


class TestValidateDataset:
    def test_valid_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(
                [{"question": "q", "answer": "a", "context": ""}],
                f,
                ensure_ascii=False,
            )
            path = Path(f.name)
        assert validate_dataset(path) is True
        path.unlink()

    def test_missing_file(self):
        assert validate_dataset(Path("nonexistent_file.json")) is False

    def test_empty_list(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump([], f)
            path = Path(f.name)
        assert validate_dataset(path) is False
        path.unlink()

    def test_missing_required_field(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump([{"question": "q"}], f)   # 缺少 answer
            path = Path(f.name)
        assert validate_dataset(path) is False
        path.unlink()


class TestLoadProcessedDataset:
    def test_load_from_local_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [
                {"question": f"q{i}", "answer": f"a{i}", "context": ""}
                for i in range(10)
            ]
            fpath = Path(tmpdir) / "hotpotqa_test.json"
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f)

            samples = load_processed_dataset(
                "hotpotqa", data_dir=tmpdir, num_samples=5
            )
            assert len(samples) == 5

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="未知数据集"):
            load_processed_dataset("unknown_dataset", data_dir="/tmp")

    def test_num_samples_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [{"question": f"q{i}", "answer": f"a{i}", "context": ""} for i in range(20)]
            fpath = Path(tmpdir) / "gsm8k_test.json"
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f)

            samples = load_processed_dataset("gsm8k", data_dir=tmpdir, num_samples=7)
            assert len(samples) == 7
