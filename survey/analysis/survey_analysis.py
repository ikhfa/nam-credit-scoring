"""Analysis of human evaluation survey results."""

import numpy as np
import pandas as pd
from scipy import stats


LIKERT_DIMENSIONS = ["clarity", "completeness", "actionability", "trust"]


def load_survey_data(path: str) -> pd.DataFrame:
    """Load survey results from CSV.

    Expected columns: participant_id, scenario_id, condition (NAM/SHAP),
    clarity, completeness, actionability, trust, preference.
    """
    return pd.read_csv(path)


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute median and IQR for each Likert dimension per condition."""
    rows = []
    for dim in LIKERT_DIMENSIONS:
        for condition in ["NAM", "SHAP"]:
            subset = df[df["condition"] == condition][dim]
            rows.append({
                "dimension": dim,
                "condition": condition,
                "median": subset.median(),
                "q25": subset.quantile(0.25),
                "q75": subset.quantile(0.75),
                "mean": subset.mean(),
                "std": subset.std(),
            })
    return pd.DataFrame(rows)


def wilcoxon_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank test on each Likert dimension (NAM vs SHAP).

    Pairs are matched by (participant_id, scenario_id).
    """
    results = []
    for dim in LIKERT_DIMENSIONS:
        nam_scores = df[df["condition"] == "NAM"].sort_values(
            ["participant_id", "scenario_id"]
        )[dim].values
        shap_scores = df[df["condition"] == "SHAP"].sort_values(
            ["participant_id", "scenario_id"]
        )[dim].values

        stat, p_value = stats.wilcoxon(nam_scores, shap_scores, alternative="two-sided")

        # Rank-biserial correlation (effect size)
        n = len(nam_scores)
        diff = nam_scores - shap_scores
        ranks = stats.rankdata(np.abs(diff))
        r_plus = np.sum(ranks[diff > 0])
        r_minus = np.sum(ranks[diff < 0])
        r_biserial = (r_plus - r_minus) / (r_plus + r_minus) if (r_plus + r_minus) > 0 else 0

        results.append({
            "dimension": dim,
            "nam_median": np.median(nam_scores),
            "shap_median": np.median(shap_scores),
            "W_statistic": stat,
            "p_value": p_value,
            "effect_size_r": r_biserial,
        })

    return pd.DataFrame(results)


def preference_test(df: pd.DataFrame) -> dict:
    """Binomial test on forced-choice preferences."""
    prefs = df.groupby(["participant_id", "scenario_id"])["preference"].first()

    n_nam = (prefs == "NAM").sum()
    n_shap = (prefs == "SHAP").sum()
    n_none = (prefs == "No difference").sum()
    total_with_pref = n_nam + n_shap

    if total_with_pref > 0:
        binom_result = stats.binom_test(n_nam, total_with_pref, 0.5)
    else:
        binom_result = 1.0

    return {
        "n_prefer_nam": int(n_nam),
        "n_prefer_shap": int(n_shap),
        "n_no_difference": int(n_none),
        "binomial_p_value": float(binom_result),
    }


def run_full_analysis(survey_path: str) -> dict:
    """Run complete survey analysis."""
    df = load_survey_data(survey_path)

    desc = descriptive_statistics(df)
    tests = wilcoxon_tests(df)
    prefs = preference_test(df)

    print("=== Descriptive Statistics ===")
    print(desc.to_string(index=False))

    print("\n=== Wilcoxon Tests ===")
    print(tests.to_string(index=False))

    print("\n=== Preference Test ===")
    for k, v in prefs.items():
        print(f"  {k}: {v}")

    return {
        "descriptive": desc,
        "wilcoxon": tests,
        "preference": prefs,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_full_analysis(sys.argv[1])
    else:
        print("Usage: python survey_analysis.py <survey_results.csv>")
