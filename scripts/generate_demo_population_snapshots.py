"""生成不含疾病标签、带连续月份时间线的机构端演示数据。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_chengdu_health_data import generate_chunk


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "demo" / "organization_snapshots"
DEFAULT_START = "2026-01"
DEFAULT_MONTHS = 6
DEFAULT_ROWS = 20_000
DEFAULT_SEED = 20260714


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--months", type=int, default=DEFAULT_MONTHS)
    parser.add_argument("--start-period", default=DEFAULT_START, help="起始月份，例如 2026-01")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _periods(start_period: str, months: int) -> list[str]:
    if not re.fullmatch(r"\d{4}-(?:0[1-9]|1[0-2])", start_period):
        raise ValueError("start-period 必须使用 YYYY-MM 格式。")
    if months <= 0 or months > 24:
        raise ValueError("months 必须在 1 到 24 之间。")
    start = pd.Period(start_period, freq="M")
    return [str(start + index) for index in range(months)]


def _mobile_phone(fake: Faker, rng: np.random.Generator) -> str:
    """优先使用 Faker 手机号，异常时用中国大陆常见号段兜底。"""
    raw = re.sub(r"\D", "", str(fake.phone_number()))
    if len(raw) == 11 and raw.startswith("1"):
        return raw
    prefixes = np.array(["130", "131", "132", "133", "135", "136", "138", "139", "150", "151", "155", "156", "157", "158", "159", "181", "186", "187", "189"])
    return str(rng.choice(prefixes)) + "".join(rng.choice(list("0123456789"), size=8))


def _build_identity(rows: int, seed: int) -> pd.DataFrame:
    fake = Faker("zh_CN")
    fake.seed_instance(seed)
    rng = np.random.default_rng(seed + 17)
    names = [fake.name() for _ in range(rows)]
    phones = [_mobile_phone(fake, rng) for _ in range(rows)]
    return pd.DataFrame(
        {
            "resident_id": np.arange(510100900001, 510100900001 + rows, dtype=np.int64),
            "name": names,
            "phone": phones,
        }
    )


def _apply_monthly_change(base: pd.DataFrame, month_index: int, months: int, seed: int) -> pd.DataFrame:
    """生成同一居民面板的月度快照，变化幅度受控且可解释。"""
    frame = base.copy()
    rng = np.random.default_rng(seed + month_index * 1009)
    progress = month_index / max(months - 1, 1)

    # 生活方式干预逐步改善，不对所有居民同时改变，避免模板化趋势。
    non_exercise = frame["exercise"].to_numpy() == 0
    exercise_switch = non_exercise & (rng.random(len(frame)) < 0.035 * month_index)
    frame.loc[exercise_switch, "exercise"] = 1

    current_smoker = frame["smoker"].to_numpy() == 2
    smoker_switch = current_smoker & (rng.random(len(frame)) < 0.018 * month_index)
    frame.loc[smoker_switch, "smoker"] = 1

    high_cholesterol = frame["cholesterol"].to_numpy() >= 2
    cholesterol_switch = high_cholesterol & (rng.random(len(frame)) < 0.012 * month_index)
    cholesterol = frame["cholesterol"].to_numpy(copy=True)
    cholesterol[cholesterol_switch] = np.maximum(
        cholesterol[cholesterol_switch].astype(int) - 1, 1
    ).astype(cholesterol.dtype)
    frame["cholesterol"] = cholesterol

    bmi = frame["bmi"].to_numpy(dtype=float)
    bmi_improvement = np.where(bmi >= 25, 0.55 * progress, 0.08 * progress)
    bmi = bmi - bmi_improvement + rng.normal(0, 0.12, len(frame))
    frame["bmi"] = np.round(np.clip(bmi, 16.0, 45.0), 2)
    frame["weight_kg"] = np.round(frame["bmi"] * (frame["height_cm"] / 100.0) ** 2, 1)

    # 血压与 BMI、运动状态保持联动，月度测量保留少量自然波动。
    exercise_relief = frame["exercise"].to_numpy() == 1
    systolic = frame["systolic_bp"].to_numpy(dtype=float)
    systolic -= progress * (1.8 * exercise_relief + 1.2 * (frame["bmi"].to_numpy() < base["bmi"].to_numpy()))
    frame["systolic_bp"] = np.rint(np.clip(systolic + rng.normal(0, 2.2, len(frame)), 90, 220)).astype(int)
    diastolic = frame["diastolic_bp"].to_numpy(dtype=float) - progress * 0.8 * exercise_relief
    frame["diastolic_bp"] = np.rint(np.clip(diastolic + rng.normal(0, 1.3, len(frame)), 50, 130)).astype(int)

    frame["synthetic_data"] = 1
    return frame


def generate_snapshots(rows: int, months: int, start_period: str, seed: int, output_dir: Path) -> list[Path]:
    periods = _periods(start_period, months)
    if rows <= 0:
        raise ValueError("rows 必须为正整数。")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 复用项目已经校准过的成都居民生理与危险因素生成逻辑，但删除所有疾病标签。
    health = generate_chunk(np.random.default_rng(seed), 0, rows).drop(
        columns=[
            "target_disease", "label_heart", "label_stroke", "label_chd",
            "label_heart_attack", "label_heart_failure", "annual_stroke_event",
            "annual_heart_attack_event",
        ],
        errors="ignore",
    )
    identity = _build_identity(rows, seed)
    base = pd.concat([identity, health.drop(columns=["resident_id"])], axis=1)

    outputs = []
    for month_index, period in enumerate(periods):
        snapshot = _apply_monthly_change(base, month_index, months, seed)
        snapshot["data_period"] = period
        snapshot["snapshot_date"] = f"{period}-01"
        output = output_dir / f"chengdu_demo_population_{period}.csv"
        snapshot.to_csv(output, index=False, encoding="utf-8-sig", float_format="%.2f")
        outputs.append(output)

    profile = {
        "purpose": "机构端连续月份趋势演示数据",
        "synthetic": True,
        "rows_per_snapshot": rows,
        "periods": periods,
        "files": [str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in outputs],
        "labels_included": False,
        "same_resident_panel": True,
        "identity_fields": ["resident_id", "name", "phone"],
        "model_features": ["age", "gender", "bmi", "cholesterol", "diabetes", "hypertension", "smoker", "alcohol", "exercise"],
        "health_generation_source": "scripts/generate_chengdu_health_data.py and mydocs/成都居民健康数据.md",
        "trend_interpretation": "不同月份是同一批仿真居民的统计快照，变化用于演示模型评分和干预观察，不代表真实因果效果。",
    }
    (output_dir / "README.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return outputs


def main() -> None:
    args = parse_args()
    outputs = generate_snapshots(args.rows, args.months, args.start_period, args.seed, args.output_dir)
    print(f"generated {len(outputs)} snapshots in {args.output_dir}")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
