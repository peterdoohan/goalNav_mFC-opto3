"""
reintroducted excess steps library for follow up analysis on group x stim effects
of increased excess steps during navigation with mFC inhibition
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import zscore, chi2, linregress, ttest_ind, ttest_1samp

from GridMaze.maze import representations as mr
from GridMaze.maze import metrics as mm
from GridMaze.maze import plotting as mp

from GridMaze.analysis.behaviour import performance_metrics as pm
from GridMaze.analysis.strategies import habits


# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

# %% Functions


def plot_excess_steps_vs_covariate(
    excess_steps_df,
    v="path_habit_score",
    stim_day_range=(6, np.inf),
    outlier_thres=500,
    x_bins=10,
    stim_color="#0077FF",
    vmax=None,
    axes=None,
):
    # set up fig
    if axes is None:
        f, axes = plt.subplots(1, 2, figsize=(4, 2), sharex=True, sharey=True)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("excess steps")
        ax.set_xlabel(v)

    # filter data
    df = _filter_excess_steps_df(excess_steps_df, stim_day_range, outlier_thres)
    if v not in df.columns:
        # add trial path antihabit score covariate
        df = add_trial_covariates(df, c=[v], zscore_vars=False)
    df.dropna(subset=[v], inplace=True)
    # bin v score for plotting
    vmax = df[v].max() if vmax is None else vmax
    df["bin"] = pd.cut(
        df[v],
        bins=np.linspace(0, vmax, x_bins + 1),
        include_lowest=True,
    )
    # average xs step per bin over subjects
    _df = df.groupby(["condition", "subject_ID", "stim_trial", "bin"], observed=True).n_excess_steps.mean()
    grouped_df = _df.groupby(level=[0, 2, 3], observed=True)  # group x stim averages
    mean = grouped_df.mean()
    sem = grouped_df.sem()
    for ax, group in zip(axes, ["control", "opto"]):
        for stim_trial, color in zip([False, True], ["dimgrey", stim_color]):
            ax.set_title(group)
            _mean = mean.loc[(group, stim_trial)]
            x = np.array([x.mid for x in _mean.index]).astype(float)
            _mean = _mean.values
            _sem = sem.loc[(group, stim_trial)].values
            ax.plot(x, _mean, color=color, label=f"stim: {stim_trial}")
            ax.fill_between(x, _mean - _sem, _mean + _sem, color=color, alpha=0.25)


# %% random effects goal difficulty tests


def plot_delta_excess_steps_fit_slopes(
    excess_steps_df,
    stim_day_range=(6, np.inf),
    outlier_thres=500,
    var="goal_fitness",
    print_stats=True,
    ax=None,
):
    """ """
    delta_df = get_stratified_delta_excess_steps_df(excess_steps_df, stim_day_range, outlier_thres, var)
    fits_df = get_random_effects_linreg_fits(delta_df, var)

    # plotting
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(1, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", lw=0.5, ls="--")

    sns.stripplot(
        data=fits_df,
        y="slope",
        hue="condition",
        palette=["black", "#0077FF"],
        alpha=0.25,
        dodge=0.01,
        legend=False,
    )
    # plot subject mean & sem
    sns.pointplot(
        data=fits_df,
        y="slope",
        hue="condition",
        palette=["black", "#0077FF"],
        errorbar="se",
        dodge=0.4,
        ax=ax,
        legend=False,
    )

    if print_stats:
        control_slopes = fits_df[fits_df.condition == "control"].slope
        opto_slopes = fits_df[fits_df.condition == "opto"].slope
        t_cond, p_cond = ttest_ind(control_slopes, opto_slopes)
        t_cont, p_cont = ttest_1samp(control_slopes, 0)
        t_opto, p_opto = ttest_1samp(opto_slopes, 0)
        print("T-tests on slope from delta xs steps linear fit:")
        print(f"  control vs opto: t({len(control_slopes) + len(opto_slopes) - 2})={t_cond:.2f}, p={p_cond:.3f}")
        print(f"  control vs 0: t({len(control_slopes) - 1})={t_cont:.2f}, p={p_cont:.3f}")
        print(f"  opto vs 0: t({len(opto_slopes) - 1})={t_opto:.2f}, p={p_opto:.3f}")

    return


def plot_stratified_delta_excess_steps(
    excess_steps_df,
    stim_day_range=(6, np.inf),
    outlier_thres=200,
    var="goal_fitness",
    maze_cmap="rainbow",
    point_labels=True,
    point_label_thres=4,
    regplot_color="silver",
    x_log=False,
    axes=None,
):
    """ """
    delta_df = get_stratified_delta_excess_steps_df(excess_steps_df, stim_day_range, outlier_thres, var)
    fits_df = get_random_effects_linreg_fits(delta_df, var)
    # plotting
    if axes is None:
        f, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True, sharex=True)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel(var)
    axes[0].set_ylabel("Δ excess steps \n (stim On - stim Off)")

    colors = sns.color_palette(maze_cmap, n_colors=49)
    for condition, ax in zip(["control", "opto"], axes):
        ax.set_title(condition)
        cond_df = delta_df[delta_df.condition == condition]
        cond_avg = cond_df.groupby(["maze_name", "goal"])[["delta_excess_steps", var]].mean()
        for maze_name, mk in zip(["maze_1", "maze_2"], ["s", "D"]):
            maze_df = cond_avg.loc[maze_name]
            # plot goal scatter
            ax.scatter(
                maze_df[var],
                maze_df.delta_excess_steps,
                label=maze_name,
                marker=mk,
                color=colors,
                s=20,
            )

            if point_labels:
                # add text labels for points with large delta excess steps
                offset = 0.02
                for goal, row in maze_df.iterrows():
                    if row.delta_excess_steps >= point_label_thres:
                        ax.text(
                            row[var] + offset,
                            row.delta_excess_steps + offset,
                            str(goal),
                            fontsize=8,
                            alpha=0.5,
                        )
        # plot linear fit (random effects x subj)
        cond_fit_df = fits_df[fits_df.condition == condition]
        mean_slope = cond_fit_df.slope.mean()
        mean_intercept = cond_fit_df.intercept.mean()
        sem_slope = cond_fit_df.slope.sem()
        sem_intercept = cond_fit_df.intercept.sem()
        x = np.linspace(delta_df[var].min(), delta_df[var].max(), 200)
        y_mean = mean_intercept + mean_slope * x
        y_sem = np.sqrt(sem_intercept**2 + (x * sem_slope) ** 2)  # SEM propagation (first-order approximation)

        # plot line and shaded error
        ax.plot(x, y_mean, color=regplot_color, lw=2)
        ax.fill_between(
            x,
            y_mean - y_sem,
            y_mean + y_sem,
            color=regplot_color,
            alpha=0.2,
            edgecolor="none",
            linewidth=0,
        )
        if x_log:
            ax.set_xscale("log")


def plot_special_maze_legend(axes=None, maze_cmap="rainbow", maze_color="silver"):
    """
    legend to go with stratified delta excess steps plot
    """
    if axes is None:
        f, axes = plt.subplots(2, 1, figsize=(2, 4))
    maze_1 = mr.get_simple_maze("maze_1")
    maze_2 = mr.get_simple_maze("maze_2")
    nodes = list(nx.get_node_attributes(maze_1, "label").values())
    colors = sns.color_palette(maze_cmap, n_colors=49)
    node2color = {n: c for n, c in zip(sorted(nodes), colors)}
    for ax, maze, mk in zip(axes, [maze_1, maze_2], ["s", "D"]):
        mp.plot_simple_maze_silhouette(
            maze,
            ax,
            color=maze_color,
            highlight_nodes=nodes,
            highlight_color=maze_color,
            special_location2color=node2color,
            node_size=35,
            edge_size=3.5,
            node_shape=mk,
        )


def get_stratified_delta_excess_steps_df(
    excess_steps_df,
    stim_day_range=(6, np.inf),
    outlier_thres=200,
    var="goal_fitness",
):
    # filter data
    df = _filter_excess_steps_df(excess_steps_df, stim_day_range, outlier_thres)

    # get delta xs steps between stim-on and stim-off per goal for each subject
    delta_df = (
        df.groupby(["condition", "subject_ID", "maze_name", "goal", "stim_trial"])
        .n_excess_steps.mean()
        .unstack(-1)
        .diff(axis=1)[True]
        .reset_index()
    )  # note: because we are stratifying by goal we no stim trials in some conditions (delta=np.nan)
    delta_df.rename(columns={True: "delta_excess_steps"}, inplace=True)
    delta_df = add_trial_covariates(delta_df, c=[var], zscore_vars=False)
    delta_df = delta_df.dropna(subset=["delta_excess_steps"])  # drop goals with no stim trials
    return delta_df


def get_random_effects_linreg_fits(delta_df, var="goal_fitness"):
    """ """
    results = []
    for condition in ["control", "opto"]:
        cond_df = delta_df[delta_df.condition == condition]
        for subject in cond_df.subject_ID.unique():
            _coefs = {"condition": condition, "subject_ID": subject}
            subj_df = cond_df[cond_df.subject_ID == subject]
            slope, intercept, *_ = linregress(subj_df[var], subj_df["delta_excess_steps"])
            _coefs["slope"] = slope
            _coefs["intercept"] = intercept
            results.append(_coefs)
    return pd.DataFrame(results)


# %% Linear mixed modelling


def run_double_3_way_linear_mixed_model(
    excess_steps_df,
    var1="goal_fitness",
    var2="goal_betweenness_centrality",
    stim_day_range=(6, np.inf),
    outlier_thres=None,
    zscore_vars=True,
    full_random_effects=True,
    print_stats_model_summaries=True,
):
    """ """
    # filter data
    df = _filter_excess_steps_df(excess_steps_df, stim_day_range, outlier_thres)

    # add covariate
    df = add_trial_covariates(df, c=[var1, var2], zscore_vars=zscore_vars)
    df.dropna(subset=[var1, var2], inplace=True)
    if full_random_effects:
        re_formula = f"~stim_trial + {var1} + {var2} + stim_trial:{var1} + stim_trial:{var2}"
    else:
        re_formula = f"~stim_trial"
    # full model (with group x stim x var1 and group x stim x var2 3-way interactions)
    md_full = smf.mixedlm(
        formula=f"n_excess_steps ~ condition * stim_trial * {var1} + condition * stim_trial * {var2}",
        data=df,
        groups=df["subject_ID"],
        re_formula=re_formula,
    )
    print("Fitting full model...")
    res_full = md_full.fit(
        reml=False, method="lbfgs", maxiter=10000
    )  # fit with maximum likelihood for model comparison

    if print_stats_model_summaries:
        print("Full model results:")
        print(res_full.summary())


def run_3way_linear_mixed_model(
    excess_steps_df,
    var="goal_fitness",
    stim_day_range=(6, np.inf),
    outlier_thres=500,
    zscore_var=True,
    print_stats_model_summaries=False,
):
    """
    Trying to ask, does a 3-way interaction with group x stim x var explain excess steps?
        Given that we have already established a strong group x stim interaction,
        does this covary with other task features?.
    """
    # filter data
    df = _filter_excess_steps_df(excess_steps_df, stim_day_range, outlier_thres)

    # add covariate
    df = add_trial_covariates(df, c=[var], zscore_vars=zscore_var)
    df.dropna(subset=[var], inplace=True)

    # full model (with group x stim x var 3-way interaction)
    md_full = smf.mixedlm(
        formula=f"n_excess_steps ~ condition * stim_trial * {var}",
        data=df,
        groups=df["subject_ID"],
        re_formula=f"~stim_trial + {var} + stim_trial:{var}",
    )
    res_full = md_full.fit(
        reml=False, method="lbfgs", maxiter=10000
    )  # fit with maximum likelihood for model comparison

    # reduced model (without group x stim x var 3-way interaction)
    md_reduced = smf.mixedlm(
        formula=f"n_excess_steps ~ (condition * stim_trial) + (condition * {var}) + (stim_trial * {var})",
        data=df,
        groups=df["subject_ID"],
        re_formula=f"~stim_trial + {var} + stim_trial:{var}",  # same random effects structure
    )
    res_reduced = md_reduced.fit(reml=False, method="lbfgs", maxiter=10000)

    # performance likelihood ratio test
    llf_full = res_full.llf
    llf_red = res_reduced.llf
    df_diff = len(res_full.fe_params) - len(res_reduced.fe_params)
    LR = 2 * (llf_full - llf_red)
    p_val = 1 - chi2.cdf(LR, df_diff)
    print("Likelihood Ratio Test for group x stim x var interaction:")
    print(f"    LR={LR:.3f}, df={df_diff}, p={p_val:.4g}")

    if print_stats_model_summaries:
        print("Full model results:")
        print(res_full.summary())
        print("\nReduced model results:")
        print(res_reduced.summary())


def add_trial_covariates(
    df,  #
    c=[
        "starting_distance_diff",
        "goal_betweenness_centrality",
        "goal_mean_geodesic_distance",
        "goal_degree",
        "goal_fitness",
        # "goal_habit_score",
    ],
    zscore_vars=True,
):
    """
    df = performance_df from GridMaze.analysis.behaviour.performance_metrics.get_performance_df()
           or equiv.
    """

    _df = df.copy()
    # check inputs
    for _c in c:
        if _c not in [
            "starting_distance_diff",
            "goal_betweenness_centrality",
            "goal_mean_geodesic_distance",
            "goal_degree",
            "goal_fitness",
            "goal_habit_score",
            "path_habit_score",
        ]:
            raise ValueError(f"Covariate {_c} not recognized.")

    maze_1 = mr.get_simple_maze("maze_1")
    maze_2 = mr.get_simple_maze("maze_2")

    def _goal2var(row, maze_1_dict, maze_2_dict):
        if row.maze_name == "maze_1":
            return maze_1_dict[row.goal]
        elif row.maze_name == "maze_2":
            return maze_2_dict[row.goal]

    if "starting_distance_diff" in c:
        # use precomputed distances from performance metrics
        v = _df["start_geodesic_dist"] - _df["start_euclidean_dist"]
        _df["starting_distance_diff"] = zscore(v) if zscore_vars else v

    if "goal_betweenness_centrality" in c:
        maze_1_dict = mm.get_betweeness_centrality(maze_1)
        maze_2_dict = mm.get_betweeness_centrality(maze_2)
        v = _df.apply(lambda row: _goal2var(row, maze_1_dict, maze_2_dict), axis=1)
        _df["goal_betweenness_centrality"] = zscore(v) if zscore_vars else v

    if "goal_mean_geodesic_distance" in c:
        maze_1_dict = mm.get_mean_geodesic_distance(maze_1)
        maze_2_dict = mm.get_mean_geodesic_distance(maze_2)
        v = _df.apply(lambda row: _goal2var(row, maze_1_dict, maze_2_dict), axis=1)
        _df["goal_mean_geodesic_distance"] = zscore(v) if zscore_vars else v

    if "goal_degree" in c:
        maze_1_dict = mm.get_node_degree(maze_1)
        maze_2_dict = mm.get_node_degree(maze_2)
        v = _df.apply(lambda row: _goal2var(row, maze_1_dict, maze_2_dict), axis=1)
        _df["goal_degree"] = zscore(v) if zscore_vars else v

    if "goal_fitness" in c:
        maze_1_dict = mm.get_fitness(maze_1)
        maze_2_dict = mm.get_fitness(maze_2)
        v = _df.apply(lambda row: _goal2var(row, maze_1_dict, maze_2_dict), axis=1)
        _df["goal_fitness"] = zscore(v) if zscore_vars else v

    if "goal_habit_score" in c:
        subject2maze2dict = {
            m: {s: habits.get_goal_habit_scores(m, s) for s in SUBJECT_IDS} for m in ["maze_1", "maze_2"]
        }

        def _get_goal_habit_score(row):
            return subject2maze2dict[row.maze_name][row.subject_ID][row.goal]

        v = _df.apply(_get_goal_habit_score, axis=1)
        _df["goal_habit_score"] = zscore(v) if zscore_vars else v

    if "path_habit_score" in c:
        subject2maze2dict = {
            m: {s: habits.get_all_pairs_habit_score(m, s) for s in SUBJECT_IDS} for m in ["maze_1", "maze_2"]
        }

        def _get_path_habit_score(row):
            start = row.first_node
            if not isinstance(start, str):
                return np.nan
            goal = row.goal
            return subject2maze2dict[row.maze_name][row.subject_ID][goal][start]

        v = _df.apply(_get_path_habit_score, axis=1)
        _df["path_habit_score"] = zscore(v, nan_policy="omit") if zscore_vars else v

    return _df


# %%


def plot_excess_steps_by_goal_heatmap_summary(
    excess_steps_df,
    maze_name="maze_2",
    stim_day_range=(6, np.inf),
    outlier_thres=500,
    vmin=None,
    vmax=10,
    plot_as="raw",
    colormap="viridis",
    axes=None,
):
    assert plot_as in ["raw", "delta", "delta_delta"]

    # filter data
    df = _filter_excess_steps_df(excess_steps_df, stim_day_range, outlier_thres)
    df = df[df.maze_name == maze_name]
    # load maze stuff
    simple_maze = mr.get_simple_maze(maze_name)
    edges = [l for l in mr.get_maze_locations(simple_maze) if "-" in l]

    if plot_as == "raw":
        # average hms across subjects in each group x stim condition
        if axes is None:
            f, axes = plt.subplots(2, 2, figsize=(6, 6))
        plot_df = df.groupby(["condition", "stim_trial", "goal"]).n_excess_steps.mean()
        _max = plot_df.max() if vmax is None else vmax
        _min = 0 if vmin is None else vmin
        for i, group in enumerate(["control", "opto"]):
            for j, stim_trial in enumerate([False, True]):
                ax = axes[i, j]
                ax.set_title(f"{group} - stim:{stim_trial}")
                hm = pd.concat([plot_df.loc[(group, stim_trial)], pd.Series(0, index=edges)])
                clabel = "excess steps"
                mp.plot_simple_heatmap(
                    simple_maze,
                    hm,
                    ax=ax,
                    colormap=colormap,
                    value_label=clabel,
                    vmin=_min,
                    vmax=_max,
                    edge_size=6,
                    node_size=150,
                )

    elif plot_as == "delta":
        # take delta stim_on - stim_off within subject, then average across groups
        if axes is None:
            f, axes = plt.subplots(1, 2, figsize=(6, 3))
        pivot_df = df.pivot_table(
            values="n_excess_steps", index=["condition", "subject_ID", "goal"], columns="stim_trial"
        )
        delta_df = pivot_df.diff(axis=1)[True].groupby(level=[0, 2]).mean()
        _min = delta_df.min() if vmin is None else vmin
        _max = delta_df.max() if vmax is None else vmax
        for ax, group in zip(axes, ["control", "opto"]):
            ax.set_title(f"{group} (stim_on - stim_off)")
            clabel = "Δ excess steps"
            hm = pd.concat([delta_df.loc[group], pd.Series(0, index=edges)])
            mp.plot_simple_heatmap(
                simple_maze,
                hm,
                ax=ax,
                colormap=colormap,
                value_label=clabel,
                vmin=_min,
                vmax=_max,
                edge_size=6,
                node_size=150,
                allow_negative=True,
            )

    elif plot_as == "delta_delta":
        # take within subject delta then aver across groups and take difference between
        # group averages for delta detla missed path rate
        if axes is None:
            f, axes = plt.subplots(1, 1, figsize=(3, 3))
        pivot_df = df.pivot_table(
            values="n_excess_steps", index=["condition", "subject_ID", "goal"], columns="stim_trial"
        )
        delta_df = pivot_df.diff(axis=1)[True].groupby(level=[0, 2]).mean()
        delta_delta = delta_df.loc["opto"] - delta_df.loc["control"]
        _min = delta_delta.min() if vmin is None else vmin
        _max = delta_delta.max() if vmax is None else vmax
        mp.plot_simple_heatmap(
            simple_maze,
            pd.concat([delta_delta, pd.Series(0, index=edges)]),
            ax=axes,
            colormap=colormap,
            value_label="ΔΔ excess steps",
            vmin=_min,
            vmax=_max,
            edge_size=6,
            node_size=150,
            allow_negative=True,
        )

    return df


# %% unitls


def _filter_excess_steps_df(
    excess_steps_df,
    stim_day_range=(6, np.inf),
    outlier_thres=500,
):
    """ """
    df = excess_steps_df.copy()
    if stim_day_range is not None:
        df = df[df.total_stim_days.between(*stim_day_range)]
    if outlier_thres is not None:
        df = df[df.n_excess_steps <= outlier_thres]
    return df


# %%
