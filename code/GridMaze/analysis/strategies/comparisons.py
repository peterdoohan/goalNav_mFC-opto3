"""
Strategy-agreement comparisons: P(target_strategy) on decision-point subsets where
named strategies agree/disagree, with a 3-way mixed LMM
(condition × stim_trial × {strats_agree | scenario | target_strategy}).
"""

# %% Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate
from pymer4.models import Lmer

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import plotting as cp
from GridMaze.analysis.strategies import get_input_data as gid

# %% Globs
MAX_STIM_DURATION = 30  # seconds

DEFAULT_SCENARIOS = {
    "all": (),
    "V=S": ("vector == structure",),
    "V≠S": ("vector != structure",),
    "H=S": ("habit == structure",),
    "H≠S": ("habit != structure",),
    "VHnotS": ("vector == habit", "vector != structure"),
    "VSnotH": ("vector == structure", "vector != habit"),
    "SHnotV": ("structure == habit", "structure != vector"),
    "all_disagree": ("habit != vector", "habit != structure", "structure != vector"),
}


# %% Functions
def plot_prop_optimal(
    navigation_strategies_df,
    strat_pair=("habit", "structure"),
    conditional_pair=None,
    conditional_agree=None,
    target_strategy="structure",
    stim_day_range=(6, gs.TOTAL_STIM_DAYS),
    stim_only=True,
    decision_point_filter="only",
    print_stats=True,
    axes=None,
):
    """Plot P(target_strategy) on agree- vs disagree-decision-points for `strat_pair`.

    Optionally restrict to a subset where `conditional_pair` agrees (or disagrees),
    e.g. conditional_pair=("vector", "habit"), conditional_agree=False ->
    only decision points where vector and habit pick different top actions.

    `target_strategy` controls what counts as a "correct" choice on the y-axis
    (default "structure" -> P(optimal); also "vector" or "habit").

    `decision_point_filter`: "all" (no filter), "only" (degree>2 nodes), or "exclude"
    (degree≤2 nodes, i.e. corridors only).
    """
    df = _filter_navigation_df(
        navigation_strategies_df, stim_day_range, stim_only, decision_point_filter, target_strategy=target_strategy
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


def plot_prop_across_scenarios(
    navigation_strategies_df,
    scenarios={
        "H≠S": ("habit != structure",),
        "V≠S": ("vector != structure",),
        "V≠S & H≠S": ("vector != structure", "habit != structure"),
    },
    target_strategy="structure",
    stim_day_range=(6, gs.TOTAL_STIM_DAYS),
    stim_only=True,
    decision_point_filter="only",
    show_chance=True,
    sharey=False,
    print_stats=True,
    axes=None,
):
    """One group×stim panel per scenario; 3-way mixed LMM (condition × stim_trial × scenario) across panels.

    `scenarios` is a dict {label: tuple_of_constraints}. Each constraint is a string of the form
    "A == B" or "A != B" where A/B are strategy names; constraints within a scenario are AND'd.
    An empty tuple means "all rows".

    `target_strategy` controls what counts as a "correct" choice on the y-axis:
    "structure" (default) -> P(optimal); "habit" -> P(matches habit); "vector" -> P(matches vector).

    `decision_point_filter`: "all" (no filter), "only" (degree>2 nodes), or "exclude"
    (degree≤2 nodes, i.e. corridors only).
    """
    df = _filter_navigation_df(
        navigation_strategies_df, stim_day_range, stim_only, decision_point_filter, target_strategy=target_strategy
    )
    scenarios = dict(scenarios)
    order = list(scenarios.keys())

    if axes is None:
        _, axes = plt.subplots(1, len(order), figsize=(2 * len(order), 3), sharey=sharey)
    axes = np.atleast_1d(axes)

    res_list = []
    for i, (label, constraints) in enumerate(scenarios.items()):
        keep_mask = _scenario_keep_mask(df, list(constraints))
        sub_df = df.loc[keep_mask]
        if print_stats:
            print(
                f"[{label}: {_scenario_label(constraints)}]: "
                f"kept {int(keep_mask.sum())}/{len(keep_mask)} decisions ({100 * keep_mask.mean():.1f}%)"
            )

        ax = axes[i]
        if len(sub_df) == 0:
            _render_empty_panel(ax, xlabel=label)
            if i == 0:
                ax.set_ylabel(f"P({target_strategy})")
            continue

        res = sub_df.groupby(["subject_ID", "condition", "stim_trial"]).correct.mean().reset_index(name="prob_correct")
        res["scenario"] = label
        res_list.append(res)

        cp.plot_group_by_stim(res, y="prob_correct", print_stats=print_stats, legend=False, ax=ax)
        ax.set_ylim(res.prob_correct.min() - 0.1, res.prob_correct.max() + 0.1)
        ax.set_xlabel(label)
        ax.set_ylabel(f"P({target_strategy})" if i == 0 else "")
        if show_chance:
            chance_mean = float(np.nanmean(_chance_match(sub_df, target_strategy)))
            ax.axhline(chance_mean, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    # 3-way mixed ANOVA: condition x stim_trial x scenario (needs ≥2 non-empty scenarios)
    if print_stats and len(res_list) >= 2:
        res_long = pd.concat(res_list, ignore_index=True)
        aov = run_3way_lmer_anova(res_long, dv="prob_correct", third_factor="scenario")
        _suptitle_3way(axes[0].figure, aov, third_factor="scenario", one_tailed=True)


def plot_match_targets(
    navigation_strategies_df,
    constraints=("vector != structure",),
    target_strategies=("vector", "structure"),
    stim_day_range=(6, gs.TOTAL_STIM_DAYS),
    stim_only=True,
    decision_point_filter="only",
    show_chance=False,
    scenario_label=None,
    print_stats=True,
    axes=None,
):
    """For decisions in a scenario subset (AND of constraints), plot a group×stim panel
    per target strategy showing P(match target).

    Each panel: cp.plot_group_by_stim of P(target) on the subset (group × stim_trial).
    Y-limits are set per panel (zoomed to that target's data range) so cross-group
    differences within a panel stay visible. show_chance overlays a dashed line per
    panel at the average P(match target | uniform-over-available) for that subset.

    `decision_point_filter`: "all" (no filter), "only" (degree>2 nodes — default), or
    "exclude" (degree≤2 nodes, i.e. corridors only).

    Runs a 3-way mixed LMM (condition × stim_trial × target_strategy) across panels.
    """
    if axes is None:
        _, axes = plt.subplots(
            1,
            len(target_strategies),
            figsize=(2 * len(target_strategies), 3),
        )
    axes = np.atleast_1d(axes)

    res_list = []
    for i, (target, ax) in enumerate(zip(target_strategies, axes)):
        df = _filter_navigation_df(
            navigation_strategies_df,
            stim_day_range,
            stim_only,
            decision_point_filter,
            target_strategy=target,
        )
        keep_mask = _scenario_keep_mask(df, list(constraints))
        sub_df = df.loc[keep_mask]

        if i == 0 and print_stats:
            print(
                f"[{_scenario_label(constraints)}]: "
                f"kept {int(keep_mask.sum())}/{len(keep_mask)} decisions ({100 * keep_mask.mean():.1f}%)"
            )

        if len(sub_df) == 0:
            _render_empty_panel(ax, xlabel=f"P({target})")
            continue

        res = sub_df.groupby(["subject_ID", "condition", "stim_trial"]).correct.mean().reset_index(name="prob_correct")
        res["target_strategy"] = target
        res_list.append(res)

        cp.plot_group_by_stim(res, y="prob_correct", print_stats=print_stats, legend=False, ax=ax)
        ax.set_ylim(res.prob_correct.min() - 0.1, res.prob_correct.max() + 0.1)
        ax.set_xlabel(f"P({target})")
        if show_chance:
            chance_mean = float(np.nanmean(_chance_match(sub_df, target)))
            ax.axhline(chance_mean, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    axes[0].set_ylabel("P(match target)")
    for ax in axes[1:]:
        ax.set_ylabel("")

    if scenario_label is not None:
        fig = axes[0].figure
        fig.subplots_adjust(bottom=0.25)
        fig.text(0.5, 0.02, scenario_label, ha="center", va="bottom", fontsize=9)

    # 3-way LMM (condition × stim_trial × target_strategy) — needs ≥2 non-empty targets
    if print_stats and len(res_list) >= 2:
        res_long = pd.concat(res_list, ignore_index=True)
        aov = run_3way_lmer_anova(res_long, dv="prob_correct", third_factor="target_strategy")
        _suptitle_3way(axes[0].figure, aov, third_factor="target_strategy", one_tailed=True)


def smoke_test_optimal(
    navigation_strategies_df=None,
    scenarios=None,
    target_strategy="structure",
    stim_day_range=(6, gs.TOTAL_STIM_DAYS),
    stim_only=True,
    decision_point_filter="only",
    show_chance=True,
    print_stats=True,
):
    """One figure: P(target_strategy) across scenarios as group×stim panels.

    Parallel to `smoke_test_match_targets` — same scenario dict, same kwargs, but here
    each scenario is a panel (target fixed) instead of a separate figure.
    """
    if navigation_strategies_df is None:
        navigation_strategies_df = gid.get_navigation_strategies_df(verbose=True)
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    print(f"\n========== P({target_strategy}) across scenarios ==========")
    plot_prop_across_scenarios(
        navigation_strategies_df,
        scenarios=scenarios,
        target_strategy=target_strategy,
        stim_day_range=stim_day_range,
        stim_only=stim_only,
        decision_point_filter=decision_point_filter,
        show_chance=show_chance,
        print_stats=print_stats,
    )


def smoke_test_match_targets(
    navigation_strategies_df=None,
    scenarios=None,
    target_strategies=("vector", "structure", "habit"),
    stim_day_range=(6, gs.TOTAL_STIM_DAYS),
    stim_only=True,
    decision_point_filter="all",
    show_chance=False,
    print_stats=True,
):
    """One figure per scenario: P(match target) across `target_strategies` as group×stim panels.

    Parallel to `smoke_test_optimal` — same scenario dict, same kwargs, but here each
    scenario gets its own figure with one panel per target_strategy.
    """
    if navigation_strategies_df is None:
        navigation_strategies_df = gid.get_navigation_strategies_df(verbose=True)
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    for label, constraints in scenarios.items():
        print(f"\n========== {label}: {_scenario_label(constraints)} ==========")
        plot_match_targets(
            navigation_strategies_df,
            constraints=constraints,
            target_strategies=target_strategies,
            stim_day_range=stim_day_range,
            stim_only=stim_only,
            decision_point_filter=decision_point_filter,
            show_chance=show_chance,
            scenario_label=f"{label}: {_scenario_label(constraints)}",
            print_stats=print_stats,
        )


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
    navigation_strategies_df, stim_day_range, stim_only, decision_point_filter, target_strategy="structure"
):
    """Apply the standard row filters and add the 'correct' column.

    'correct' is True when the subject's chosen action is among the top-ranked actions of
    `target_strategy` (tie- and availability-aware). target_strategy='structure' uses the
    precomputed `optimal_action` column (mathematically equivalent to argmax of structure
    values restricted to available actions).

    `decision_point_filter`: "all" (no degree filter), "only" (degree > 2 nodes — true
    decision points), or "exclude" (degree ≤ 2 nodes — corridors only).
    """
    df = navigation_strategies_df.copy()
    if stim_day_range is not None:
        df = df[df.total_stim_days.between(*stim_day_range)]
    if stim_only:
        df = df[df.time_in_trial.le(MAX_STIM_DURATION)]
    n_avail = df.available.sum(axis=1)
    if decision_point_filter == "only":
        df = df[n_avail.gt(2)]
    elif decision_point_filter == "exclude":
        df = df[n_avail.le(2)]
    elif decision_point_filter != "all":
        raise ValueError(f"decision_point_filter must be 'only', 'exclude', or 'all'; got {decision_point_filter!r}")
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


def _scenario_label(constraints):
    """Pretty label for an iterable of constraints; '()' -> 'all'."""
    return " & ".join(constraints) if constraints else "all"


def _render_empty_panel(ax, xlabel):
    """Placeholder rendering for an empty-data panel (e.g. degree-≤2 nodes have no
    rows where two strategies disagree)."""
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="grey")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(xlabel)


def _chance_match(df, target_strategy):
    """Per-row P(match target_strategy | uniform random over available actions).

    Equals |top_actions(target) ∩ available| / |available|. Adapts to ties and node degree.
    """
    if target_strategy == "structure":
        top = (df.optimal_action == 1).values
    else:
        top = _tied_argmax_mask(df[target_strategy], available_df=df.available)
    avail = df.available.values.astype(bool)
    n_top = (top & avail).sum(axis=1)
    n_avail = avail.sum(axis=1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        chance = np.where(n_avail > 0, n_top / n_avail, np.nan)
    return chance


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
