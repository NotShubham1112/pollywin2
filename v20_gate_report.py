"""Pure gate-evaluation + submission-writer functions for the v20 gate.

compute_gate_report() decides whether the self-trained SMILES encoder arm
(v20) may replace the P14 final submission. write_submission() emits the
final CSV in the exact P14 format (vault/final_synthesis.py): a single
id,target frame, 4940 rows, id order = test.csv, index=False.
"""

import pandas as pd

# Pre-registered gate knobs (see task brief; do not soften).
THR_MEAN = 0.003
THR_WORST = 0.003
ALPHA_CAP = 0.30

SUBMISSION_ROWS = 4940


def compute_gate_report(mean_delta, worst_delta, alphas,
                        thr_mean=THR_MEAN, thr_worst=THR_WORST,
                        alpha_cap=ALPHA_CAP):
    """Evaluate the v20 gate.

    Parameters
    ----------
    mean_delta : float
        mean_v20 - P14 reference mean R^2 (0.8641).
    worst_delta : float
        min over targets of (r2_v20[t] - r2_p14[t]).
    alphas : sequence of float
        Per-target blend regularization alphas chosen by blend_3d.
    thr_mean, thr_worst, alpha_cap : float
        Gate thresholds. Defaults are the pre-registered values.

    Returns
    -------
    dict with keys: pass, mean_delta, worst_delta, alphas_ok.
    pass = (mean_delta >= thr_mean) AND (worst_delta >= -thr_worst)
           AND every alpha <= alpha_cap.
    """
    alphas_ok = all(float(a) <= alpha_cap for a in alphas)
    mean_ok = float(mean_delta) >= thr_mean
    worst_ok = float(worst_delta) >= -thr_worst
    return {
        "pass": bool(mean_ok and worst_ok and alphas_ok),
        "mean_delta": float(mean_delta),
        "worst_delta": float(worst_delta),
        "alphas_ok": bool(alphas_ok),
    }


def write_submission(df, path):
    """Write the v20 submission in P14 format.

    Parameters
    ----------
    df : pd.DataFrame with columns id, target; exactly 4940 rows, id order
         matching official_dataset/test.csv.
    path : str or os.PathLike — destination CSV path.

    Returns
    -------
    path passed in (for chaining).

    Raises
    ------
    ValueError if the frame does not have exactly 4940 rows or is missing
    the id/target columns.
    """
    if list(df.columns) != ["id", "target"]:
        raise ValueError(
            f"submission frame must have columns ['id','target'], got "
            f"{list(df.columns)}")
    if len(df) != SUBMISSION_ROWS:
        raise ValueError(
            f"submission must have exactly {SUBMISSION_ROWS} rows, got {len(df)}")
    df.to_csv(path, index=False)
    return path