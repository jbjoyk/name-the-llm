"""
Offline submission validation script.
Tests SubmissionPipeline against 10, 100, 1,000 synthetic rows and 15 extreme edge cases.
"""

import os, sys, time
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.submission_pipeline import SubmissionPipeline
from scripts.validate_submission import validate_submission_df

def run_tests():
    print("=" * 70)
    print("PHASE 9D: OFFLINE SUBMISSION KERNEL VALIDATION")
    print("=" * 70)
    
    train_path = os.path.join(BASE_DIR, "data/train.csv")
    pool_path = os.path.join(BASE_DIR, "data/external/open_pool_clean.parquet")
    
    train_df = pd.read_csv(train_path)
    pool_df = pd.read_parquet(pool_path)
    
    # Sample balanced Open Pool (1,000 per class for comprehensive coverage)
    balanced_pool = pd.concat([
        g.sample(min(len(g), 1000), random_state=42) 
        for _, g in pool_df.groupby("model")
    ], ignore_index=True)
    
    print(f"Loaded {len(train_df)} official rows and {len(balanced_pool)} balanced Open Pool rows.")
    
    # Initialize and fit pipeline
    pipeline = SubmissionPipeline(config={"w_gauss": 0.05, "w_pool": 0.35, "svc_c": 0.5})
    t_fit_start = time.time()
    pipeline.fit(train_df, balanced_pool)
    fit_time = time.time() - t_fit_start
    print(f"Pipeline fitted in {fit_time:.2f} seconds.")
    
    # ----------------------------------------------------
    # Suite 1: Malformed and Edge-Case Test (15 rows)
    # ----------------------------------------------------
    edge_cases = pd.DataFrame({
        "id": [f"edge_{i:02d}" for i in range(15)],
        "text": [
            "",                                          # 0: Empty string
            "   \n\t\n   ",                              # 1: Whitespace only
            "Hello world!",                              # 2: Trivial short sentence
            "✨ 🚀 🔥 🤖 🧠 🧬 🧪 🔬 📊 📈",              # 3: Pure emojis
            "Привет мир! 这是中文测试。 مرحبا بالعالم",    # 4: Multilingual non-ASCII
            "def fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)", # 5: Code block
            "# Header\n\n- Item 1\n- Item 2\n\n> Blockquote\n```python\nx = 1\n```", # 6: Markdown
            "A" * 25000,                                 # 7: Extreme length (25,000 chars)
            None,                                        # 8: Python None
            np.nan,                                      # 9: Float NaN
            "1234567890 9876543210 1122334455",          # 10: Pure digits
            "---\n===\n***\n___\n~~~",                   # 11: Punctuation fences
            "What is quantum computing? Explain in detail.", # 12: Standard prompt-like
            "In conclusion, the results demonstrate that...", # 13: Academic conclusion
            "As an AI language model, I do not have feelings." # 14: AI disclaimer
        ]
    })
    
    print("\n[Suite 1] Testing 15 Malformed Edge Cases...")
    t0 = time.time()
    sub_edge = pipeline.predict(edge_cases)
    t_edge = time.time() - t0
    report_edge = validate_submission_df(sub_edge, edge_cases)
    
    print(f"Edge Cases Execution: {t_edge:.3f}s | Rows: {len(sub_edge)} | Valid: {report_edge['valid']}")
    for i, row in sub_edge.iterrows():
        print(f"  {row['id']}: -> {row['prediction']}")
        
    assert report_edge["valid"], f"Edge cases failed validation: {report_edge['errors']}"
    assert len(sub_edge) == 15, "Edge cases output row count mismatch"
    assert not sub_edge["prediction"].isna().any(), "Edge cases produced NaN predictions"
    print(">> Suite 1 PASSED without crashes or invalid predictions.")
    
    # ----------------------------------------------------
    # Suite 2: Scale Tests (10, 100, 1,000 rows)
    # ----------------------------------------------------
    scale_benchmarks = []
    for n in [10, 100, 1000]:
        print(f"\n[Suite 2] Testing Scale N = {n} rows...")
        np.random.seed(42 + n)
        sampled_texts = np.random.choice(train_df["text"].values, size=n, replace=True)
        test_df = pd.DataFrame({
            "id": [f"test_{n}_{i:04d}" for i in range(n)],
            "text": sampled_texts
        })
        
        t0 = time.time()
        sub_df = pipeline.predict(test_df)
        elapsed = time.time() - t0
        throughput = n / max(elapsed, 1e-6)
        
        report = validate_submission_df(sub_df, test_df)
        print(f"  N = {n:4d}: Valid={report['valid']}, Duration={elapsed:.2f}s, Throughput={throughput:.1f} rows/s, Classes={report['predicted_classes_count']}/18")
        
        assert report["valid"], f"Scale test N={n} failed: {report['errors']}"
        assert len(sub_df) == n, f"Row count mismatch for N={n}"
        assert not sub_df["prediction"].isna().any(), f"NaN detected for N={n}"
        assert sub_df["id"].tolist() == test_df["id"].tolist(), f"ID mismatch for N={n}"
        
        scale_benchmarks.append({
            "n_rows": n,
            "duration_sec": elapsed,
            "throughput_rows_per_sec": throughput,
            "classes_predicted": report["predicted_classes_count"]
        })
        
    print("\n" + "=" * 70)
    print("ALL SUBMISSION VALIDATION TESTS PASSED (100% SUCCESS)")
    print("=" * 70)
    return {
        "fit_time": fit_time,
        "edge_cases_passed": report_edge["valid"],
        "scale_benchmarks": scale_benchmarks
    }

if __name__ == "__main__":
    run_tests()
