"""
Strategy-agreement comparisons: P(target_strategy) on decision-point subsets where
named strategies agree/disagree, with a 3-way mixed LMM (condition × stim_trial × {strats_agree | scenario}).
"""

# %% Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate
from pymer4.models import Lmer

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import plotting as cp

# %% Globs
MAX_STIM_DURATION = 30  # seconds


# %% Functions
def plot_prop_optimal(
    navigation_strategies_df,
    strat_pair=("habit", "structure"),
    conditional_pair=None,
    conditional_agree=None,
    target_strategy="structure",
    stim_day_range=(6, gs.TOTAL_STIM_DAYS),
    stim_only=True,
    decision_points_only=False,
    print_stats=True,
    axes=None,
):
    """Plot P(target_strategy) on agree- vs disagree-decision-points for `strat_pair`.

    Optionally restrict to a subset where `conditional_pair` agrees (or disagrees),
    e.g. conditional_pair=("vector", "habit"), conditional_agree=False ->
    only decision points where vector and habit pick different top actions.

    `target_strategy` controls what counts as a "correct" choice on the y-axis
    (default "structure" -> P(optimal); also "vector" or "habit").
    """
    df = _filter_navigation_df(
        navigation_strategies_df, stim_day_range, stim_only, decision_points_only, target_strategy=target_strategy
    )

    # apply conditional filter (e.g. only rows where vec == habit)
    if conditional_pair is not None:
        if conditional_agree is None:
            raise ValueError("conditional_agree must be True or False when conditional_pair is set")
        c1, c2 = conditional_pair
        cond_agree = _strats_agree_mask(df, c1, c2)
        cond_keep = cond_agree if conditional_agree else ~cond_agree
        if print_stats:
            cond_rel = "==" if conditional_agree else "!="
            print(
                f"Conditional {c1} {cond_rel} {c2}: "
                f"kept {int(cond_keep.sum())}/{len(cond_keep)} decisions ({100 * cond_keep.mean():.1f}%)"
            )
        df = df.loc[cond_keep]

    # set up fig
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(4, 3))

    strat_1, strat_2 = strat_pair
    agree_mask = _strats_agree_mask(df, strat_1, strat_2)

    # label suffix for the conditional, used in prints + xlabels
    if conditional_pair is not None:
        cond_rel = "==" if conditional_agree else "!="
        cond_suffix = f" | {conditional_pair[0]} {cond_rel} {conditional_pair[1]}"
        xlabel_suffix = f"\n| {conditional_pair[0]} {cond_rel} {conditional_pair[1]}"
    else:
        cond_suffix = ""
        xlabel_suffix = ""

    # plot agree and disagree side by side
    res_list = []
    for i, (strats_agree, ax) in enumerate(zip([True, False], axes)):
        keep_mask = agree_mask if strats_agree else ~agree_mask
        sub_df = df.loc[keep_mask]
        if print_stats:
            rel_word = "agree" if strats_agree else "disagree"
            print(f"{strat_1} {rel_word} {strat_2}{cond_suffix}: included {100 * keep_mask.mean():.1f}% of decisions")
        res = sub_df.groupby(["subject_ID", "condition", "stim_trial"]).correct.mean().reset_index(name="prob_correct")
        res["strats_agree"] = strats_agree
        res_list.append(res)
        cp.plot_group_by_stim(res, y="prob_correct", print_stats=print_stats, legend=False, ax=ax)
        ax.set_ylim(res.prob_correct.min() - 0.1, res.prob_correct.max() + 0.1)
        rel = "==" if strats_agree else "!="
        ax.set_xlabel(f"{strat_1} \n {rel} \n {strat_2}{xlabel_suffix}")
        ax.set_ylabel(f"P({target_strategy})" if i == 0 else "")

    # 3-way mixed ANOVA: condition x stim_trial x strats_agree (on the conditional subset)
    if print_stats:
        res_long = pd.concat(res_list, ignore_index=True)
        aov = run_3way_lmer_anova(res_long, dv="prob_correct", third_factor="strats_agree")
        _suptitle_3way(axes[0].figure, aov, third_factor="strats_agree", one_tailed=True)


def plot_prop_scenarios(
    navigation_strategies_df,
    scenario_1=["vector == structure", "vector != habit", "structure != habit"],
    scenario_2=["vector != structure", "vector != habit", "structure != habit"],
    target_strategy="structure",
    stim_day_range=(6, gs.TOTAL_STIM_DAYS),
    stim_only=True,
    decision_points_only=False,
    print_stats=True,
    axes=None,
):
    """Plot P(target_strategy) on two scenarios defined as conjunctions of strategy-agreement constraints.

    Each scenario is a list of constraint strings of the form "A == B" or "A != B" where A/B are
    strategy names (e.g. "vector", "structure", "habit"). All constraints in a scenario are AND'd
    together to define the included decision points for that scenario's panel.

    `target_strategy` controls what counts as a "correct" choice on the y-axis:
    "structure" (default) -> P(optimal); "habit" -> P(matches habit); "vector" -> P(matches vector).

    Runs the same condition x stim_trial x scenario 3-way LMM as plot_prop_optimal.
    """
    df = _filter_navigation_df(
        navigation_strategies_df, stim_day_range, stim_only, decision_points_only, target_strategy=target_strategy
    )

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(4, 3))

    res_list = []
    for i, (constraints, ax) in enumerate(zip([scenario_1, scenario_2], axes)):
        keep_mask = _scenario_keep_mask(df, constraints)
        sub_df = df.loc[keep_mask]
        if print_stats:
            print(
                f"[{' & '.join(constraints)}]: "
                f"kept {int(keep_mask.sum())}/{len(keep_mask)} decisions ({100 * keep_mask.mean():.1f}%)"
            )
        res = sub_df.groupby(["subject_ID", "condition", "stim_trial"]).correct.mean().reset_index(name="prob_correct")
        res["scenario"] = f"scenario_{i + 1}"
        res_list.append(res)
        cp.plot_group_by_stim(res, y="prob_correct", print_stats=print_stats, legend=False, ax=ax)
        ax.set_ylim(res.prob_correct.min() - 0.1, res.prob_correct.max() + 0.1)
        ax.set_xlabel("\n".join(constraints))
        ax.set_ylabel(f"P({target_strategy})" if i == 0 else "")

    # 3-way mixed ANOVA: condition x stim_trial x scenario
    if print_stats:
        res_long = pd.concat(res_list, ignore_index=True)
        aov = run_3way_lmer_anova(res_long, dv="prob_correct", third_factor="scenario")
        _suptitle_3way(axes[0].figure, aov, third_factor="scenario", one_tailed=True)


def run_3way_lmer_anova(res_long, dv="prob_correct", third_factor="strats_agree"):
    """3-way mixed ANOVA via lmer: condition x stim_trial x {third_factor}.

    Type III F-table with Satterthwaite df, computed by R's lmerTest via pymer4.
    Random intercept per subject; condition is between-subject and is handled correctly
    by the lme4 formula (it does not vary within subject_ID).
    """
    formula = f"{dv} ~ condition * stim_trial * {third_factor} + (1|subject_ID)"
    m = Lmer(formula, data=res_long)
    m.fit(summarize=False)
    aov = m.anova()
    print(f"3-way mixed ANOVA (condition × stim_trial × {third_factor}, lmer, Type III, Satterthwaite df):")
    print(tabulate(aov, headers="keys", tablefmt="psql"))
    return aov


def _filter_navigation_df(
    navigation_strategies_df, stim_day_range, stim_only, decision_points_only, target_strategy="structure"
):
    """Apply the standard row filters and add the 'correct' column.

    'correct' is True when the subject's chosen action is among the top-ranked actions of
    `target_strategy` (tie- and availability-aware). target_strategy='structure' uses the
    precomputed `optimal_action` column (mathematically equivalent to argmax of structure
    values restricted to available actions).
    """
    df = navigation_strategies_df.copy()
    if stim_day_range is not None:
        df = df[df.total_stim_days.between(*stim_day_range)]
    if stim_only:
        df = df[df.time_in_trial.le(MAX_STIM_DURATION)]
    if decision_points_only:
        # nodes with degree 3 or 4
        df = df[df.available.sum(axis=1).gt(2)]
    chosen = (df.subject_choice == 1).values
    if target_strategy == "structure":
        top_mask = (df.optimal_action == 1).values
    else:
        top_mask = _tied_argmax_mask(df[target_strategy], available_df=df.available)
    df[("correct", "")] = (chosen & top_mask).any(axis=1)
    return df


def _strats_agree_mask(df, strat_a, strat_b):
    """Boolean Series: rows where strat_a and strat_b share an argmax action (tie- and availability-aware)."""
    a = _tied_argmax_mask(df[strat_a], available_df=df.available)
    b = _tied_argmax_mask(df[strat_b], available_df=df.available)
    return pd.Series((a & b).any(axis=1), index=df.index)


def _parse_constraint(constraint):
    """'A == B' -> ('A', True, 'B'); 'A != B' -> ('A', False, 'B')."""
    if "==" in constraint:
        a, b = constraint.split("==")
        return a.strip(), True, b.strip()
    if "!=" in constraint:
        a, b = constraint.split("!=")
        return a.strip(), False, b.strip()
    raise ValueError(f"Cannot parse constraint: {constraint!r}. Expected 'A == B' or 'A != B'.")


def _scenario_keep_mask(df, constraints):
    """AND together a list of agree/disagree constraints into a single keep mask."""
    keep = pd.Series(True, index=df.index)
    for c in constraints:
        a, want_agree, b = _parse_constraint(c)
        agree = _strats_agree_mask(df, a, b)
        keep &= agree if want_agree else ~agree
    return keep


def _suptitle_3way(fig, aov, third_factor="strats_agree", one_tailed=False):
    """Set a 2-line suptitle with the 3-way interaction F, df, p."""
    row_name = f"condition:stim_trial:{third_factor}"
    if row_name not in aov.index:
        return
    r = aov.loc[row_name]
    p = r["P-val"] / 2 if one_tailed else r["P-val"]
    fig.suptitle(
        f"3-way (group × stim × {third_factor}):\n"
        f"F({r['NumDF']:.0f}, {r['DenomDF']:.1f}) = {r['F-stat']:.2f}, p = {p:.3g}",
        fontsize=9,
    )


def _tied_argmax_mask(action_values_df, available_df=None):
    """Return a boolean (n_rows x n_actions) mask for all tied argmax actions.
    If available_df is provided, unavailable actions are excluded from the argmax.
    """

    values = action_values_df.to_numpy(dtype=float)
    avail = available_df.values.astype(bool)
    values = np.where(avail, values, -np.inf)

    row_max = np.nanmax(values, axis=1)
    tied = np.isclose(values, row_max[:, None], rtol=0.0, atol=0.0, equal_nan=False)
    return tied
