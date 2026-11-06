#!/usr/bin/env python3
"""
Submission validation script for Kaggle Name That LLM competition.
Validates submission CSV format, schema, column names, class labels, and null values.
"""

import os, sys, argparse
import pandas as pd
import numpy as np

OFFICIAL_CLASSES = [
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
    'gpt-6-astra', 'kimi-k3', 'qwen3.8-max'
]


def validate_submission_df(sub_df, test_df=None):
    """Validates submission DataFrame against competition rules."""
    errors = []
    warnings = []
    
    # 1. Schema check
    required_cols = ["id", "prediction"]
    if list(sub_df.columns) != required_cols:
        errors.append(f"Invalid columns: expected {required_cols}, found {list(sub_df.columns)}")
        
    # 2. Row count check
    if test_df is not None:
        if len(sub_df) != len(test_df):
            errors.append(f"Row count mismatch: submission has {len(sub_df)} rows, test has {len(test_df)} rows")
        if "id" in sub_df.columns and "id" in test_df.columns:
            if not (sub_df["id"].values == test_df["id"].values).all():
                errors.append("IDs do not match test IDs in exact order")
                
    # 3. Null / NaN check
    null_counts = sub_df.isna().sum().to_dict()
    if any(c > 0 for c in null_counts.values()):
        errors.append(f"Null values detected: {null_counts}")
        
    # 4. Valid class strings
    if "prediction" in sub_df.columns:
        invalid_classes = set(sub_df["prediction"].unique()) - set(OFFICIAL_CLASSES)
        if len(invalid_classes) > 0:
            errors.append(f"Invalid class labels detected: {invalid_classes}")
            
        # Class distribution check
        unique_predicted = len(sub_df["prediction"].unique())
        if len(sub_df) >= 18 and unique_predicted < 5:
            warnings.append(f"Low diversity: only {unique_predicted} distinct classes predicted out of 18")
            
    # 5. Duplicate ID check
    if "id" in sub_df.columns:
        dup_count = sub_df["id"].duplicated().sum()
        if dup_count > 0:
            errors.append(f"Duplicate IDs detected: {dup_count} duplicates")
            
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "row_count": len(sub_df),
        "predicted_classes_count": len(sub_df["prediction"].unique()) if "prediction" in sub_df.columns else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Validate Name That LLM submission CSV.")
    parser.add_argument("submission_path", help="Path to submission CSV")
    parser.add_argument("--test_path", default=None, help="Optional path to test CSV for alignment check")
    args = parser.parse_args()
    
    if not os.path.exists(args.submission_path):
        print(f"ERROR: File not found: {args.submission_path}")
        sys.exit(1)
        
    sub_df = pd.read_csv(args.submission_path)
    test_df = pd.read_csv(args.test_path) if args.test_path and os.path.exists(args.test_path) else None
    
    report = validate_submission_df(sub_df, test_df)
    
    print("\n" + "=" * 60)
    print("SUBMISSION VERIFICATION REPORT")
    print("=" * 60)
    print(f"File:           {args.submission_path}")
    print(f"Row Count:      {report['row_count']}")
    print(f"Class Diversity:{report['predicted_classes_count']} / 18 unique classes")
    print(f"Status:         {'PASSED' if report['valid'] else 'FAILED'}")
    
    if report['warnings']:
        print("\nWarnings:")
        for w in report['warnings']:
            print(f"  [!] {w}")
            
    if report['errors']:
        print("\nErrors:")
        for e in report['errors']:
            print(f"  [X] {e}")
        print("=" * 60)
        sys.exit(1)
    else:
        print("\nAll submission requirements verified successfully!")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
