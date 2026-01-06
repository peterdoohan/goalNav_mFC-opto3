"""
Plotting functions common to many analyses
@peterdoohan
"""

# %% Imports
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from pingouin import mixed_anova

# %% Global Variables


# %% Functions


def plot_group_by_stim(df, y, ax=None, stim_color="#0077FF", print_stats=False):
    """ """
    # set up fig
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 3))
    conditions = ["control", "opto"]
    x_pos = {cond: i for i, cond in enumerate(conditions)}
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([x_pos[c] for c in conditions])
    ax.set_xticklabels(conditions)
    ax.set_xlim(-0.4, len(conditions) - 0.6)
    ax.set_ylim(df[y].min() * 0.8, df[y].max() * 1.1)
    condition2color = {"control": "black", "opto": stim_color}

    # plot subject-level paired points
    for cond in conditions:
        cond_df = df[df["condition"] == cond]
        cond_df = cond_df.set_index(["subject_ID", "stim_trial"])[y].unstack()
        x_off = x_pos[cond] - 0.30
        x_on = x_pos[cond] + 0.30
        for subj, row in cond_df.iterrows():
            y_off = row[False]
            y_on = row[True]
            ax.plot([x_off, x_on], [y_off, y_on], "-", color="lightgrey", lw=1.5, alpha=0.8)

    # plot cross subject mean ± SEM
    sns.pointplot(
        data=df,
        x="condition",
        order=conditions,
        y=y,
        hue="stim_trial",
        dodge=0.3,
        linestyle="none",
        errorbar="se",
        palette=[condition2color[c] for c in conditions],
        ax=ax,
    )
    sns.move_legend(
        ax,
        "lower center",
        ncol=2,
        title="light on",
        frameon=True,
        fontsize="x-small",
    )
    ax.set_xlabel("group")
    ax.set_ylabel(y)
    ax.set_ylim(bottom=0)
    if print_stats:
        stats_df = mixed_anova(
            dv=y,
            within="stim_trial",
            between="condition",
            subject="subject_ID",
            data=df,
        )
        # extract and display relevant stats
        cond = stats_df.loc[stats_df["Source"] == "condition"].iloc[0]
        stim = stats_df.loc[stats_df["Source"] == "stim_trial"].iloc[0]
        inter = stats_df.loc[stats_df["Source"] == "Interaction"].iloc[0]
        textstr = (
            f"Group: p={cond['p-unc']:.3f}\n" f"Stim  : p={stim['p-unc']:.3f}\n" f"Int     : p={inter['p-unc']:.3f}\n"
        )
        ax.text(
            0.05,
            0.80,
            textstr,
            transform=ax.transAxes,
            fontsize=8,
        )
        print(f"Mixed ANOVA: {y}")
        print(tabulate(stats_df, headers="keys", tablefmt="psql", showindex=False))
