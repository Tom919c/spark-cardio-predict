"""
Unified feature pipeline service.

Orchestrates the full local data cleaning flow:
  raw/datasets/ (6 sources) → staging/CVD_Standard_DWD.csv
"""
import os

from src.ml.feature_engineering import build_dwd_dataset


class FeatureService:
    """Runs the local DWD data cleaning pipeline."""

    def __init__(self, raw_dir=None, staging_dir=None, output_filename="CVD_Standard_DWD.csv"):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.raw_dir = raw_dir or os.path.join(base_dir, "data", "raw", "datasets")
        self.staging_dir = staging_dir or os.path.join(base_dir, "data", "staging")
        self.output_path = os.path.join(self.staging_dir, output_filename)

    def run_pipeline(self):
        """Execute full pipeline: load → merge → clean → impute → save."""
        print("=" * 56)
        print("  FeatureService — DWD data cleaning pipeline")
        print("=" * 56)
        df = build_dwd_dataset(self.raw_dir)
        os.makedirs(self.staging_dir, exist_ok=True)
        df.to_csv(self.output_path, index=False, encoding="utf-8-sig")
        size_mb = os.path.getsize(self.output_path) / (1024 * 1024)
        print("-" * 56)
        print(f"  Saved : {self.output_path}")
        print(f"  Rows  : {len(df):,}  |  Columns : {len(df.columns)}  |  Size : {size_mb:.1f} MB")
        print(f"  label_heart=1 : {int(df['label_heart'].sum()):,}")
        print(f"  label_stroke=1: {int(df['label_stroke'].sum()):,}")
        print("=" * 56)
        return {
            "output_path": self.output_path,
            "rows": len(df),
            "columns": len(df.columns),
            "label_heart_positive": int(df["label_heart"].sum()),
            "label_stroke_positive": int(df["label_stroke"].sum()),
        }
