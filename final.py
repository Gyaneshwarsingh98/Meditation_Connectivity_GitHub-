# ============================================================
# Alpha-band wPLI Functional Connectivity Analysis
# Final, Frozen Script for Manuscript Results
# ============================================================

# -------------------------------
# 0. Imports
# -------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# -------------------------------
# 1. Configuration
# -------------------------------
DATA_PATH = "/Users/gyaneshwarsingh/Brain_connectivity/Final_dataset/Brain_connectivity.csv"

OUT_DIR = "alpha_wpli_results_2"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_BAND = "Alpha"
TARGET_METHODS = ["wpli"]

PHASE_ORDER = ["B", "M1", "M2", "T1", "T2", "P1", "P2"]

KEEP_NETWORKS = [
    "Frontal-Parietal",
    "Frontal-Temporal",
    "Intra-Frontal",
    "Intra-Parietal"
]

# -------------------------------
# 2. Load data
# -------------------------------
df = pd.read_csv(DATA_PATH)
print("Raw data shape:", df.shape)
print(df.head())
print(df['Lobe_1'].unique())
print(df['Lobe_2'].unique())
# -------------------------------
# 3. Filter to alpha-band wPLI
# -------------------------------
df_filt = (
    df
    .query("Band == @TARGET_BAND")
    .query("Method in @TARGET_METHODS")
    .copy()
)

print("Filtered data shape:", df_filt.shape)

# -------------------------------
# 4. Define lobar network classes
# -------------------------------
def network_class(row):
    if row["Lobe_1"] == row["Lobe_2"]:
        return f"Intra-{row['Lobe_1']}"
    else:
        lobes = sorted([row["Lobe_1"], row["Lobe_2"]])
        return f"{lobes[0]}-{lobes[1]}"

df_filt["Network"] = df_filt.apply(network_class, axis=1)

print(sorted(df_filt["Network"].unique()))
print(df_filt["Network"].value_counts())

df_filt = df_filt[df_filt["Network"].isin(KEEP_NETWORKS)]
print("After network restriction:", df_filt.shape)

# -------------------------------
# 5. Aggregate edge-level → subject-level
# -------------------------------
agg_df = (
    df_filt
    .groupby(
        ["Subject", "Group", "Phase", "Method", "Network"],
        observed=True
    )
    .agg(
        mean_connectivity=("Value", "mean"),
        median_connectivity=("Value", "median"),
        sd_connectivity=("Value", "std"),
        n_edges=("Value", "count")
    )
    .reset_index()
)

# -------------------------------
# 6. Prepare modeling table
# -------------------------------
model_df = (
    agg_df
    .loc[:, ["Subject", "Group", "Phase", "Method", "Network", "mean_connectivity"]]
    .rename(columns={"mean_connectivity": "Connectivity"})
)

# Categorical encoding
model_df["Subject"] = model_df["Subject"].astype("category")
model_df["Group"] = model_df["Group"].astype("category")
model_df["Network"] = model_df["Network"].astype("category")
model_df["Method"] = model_df["Method"].astype("category")

model_df["Phase"] = (
    model_df["Phase"]
    .astype("category")
    .cat.set_categories(PHASE_ORDER, ordered=True)
)

# Save modeling table (CRITICAL)
model_df.to_csv(
    os.path.join(OUT_DIR, "alpha_wpli_model_table.csv"),
    index=False
)

# -------------------------------
# 7. Linear Mixed-Effects Model
# -------------------------------
formula = "Connectivity ~ Phase * Group + Network"

lmm = smf.mixedlm(
    formula,
    data=model_df,
    groups=model_df["Subject"]
)

result = lmm.fit(reml=True)
print(result.summary())

# Save full model summary
with open(os.path.join(OUT_DIR, "alpha_wpli_lmm_summary.txt"), "w") as f:
    f.write(result.summary().as_text())

# -------------------------------
# Save coefficients table (MixedLM-safe)
# -------------------------------

coef_df = pd.DataFrame({
    "Term": result.params.index,
    "Coef": result.params.values,
    "Std_Err": result.bse.values,
    "z": result.tvalues.values,
    "p_value": result.pvalues.values
})

# Confidence intervals
ci = result.conf_int()
coef_df["CI_lower"] = ci[0].values
coef_df["CI_upper"] = ci[1].values

coef_df.to_csv(
    os.path.join(OUT_DIR, "alpha_wpli_lmm_coefficients.csv"),
    index=False
)


# -------------------------------
# 8. Descriptive summaries (for figures only)
# -------------------------------
emm = (
    model_df
    .groupby(["Group", "Phase", "Network"], observed=True)
    ["Connectivity"]
    .mean()
    .reset_index()
)

emm.to_csv(
    os.path.join(OUT_DIR, "alpha_wpli_emm_group_phase_network.csv"),
    index=False
)

network_means = (
    model_df
    .groupby(["Network"], observed=True)
    ["Connectivity"]
    .mean()
    .reset_index()
)
print(model_df["Network"].value_counts())
network_means.to_csv(
    os.path.join(OUT_DIR, "alpha_wpli_network_means.csv"),
    index=False
)

phase_means = (
    model_df
    .groupby(["Phase"], observed=True)
    ["Connectivity"]
    .mean()
    .reset_index()
)

phase_means.to_csv(
    os.path.join(OUT_DIR, "alpha_wpli_phase_means.csv"),
    index=False
)

# -------------------------------
# -------------------------------
# 9. Publication-ready visualization (revised)
# -------------------------------

FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 300
})

GROUP_COLORS = {
    "CG": "#4C4C4C",
    "STM": "#1f77b4",
    "LTM": "#d62728"
}

# ============================================================
# Figure 2 — Phase × Group with CI ribbons
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

summary = (
    model_df
    .groupby(["Group", "Phase"], observed=True)
    .Connectivity
    .agg(["mean", "sem"])
    .reset_index()
)

for group, gdf in summary.groupby("Group"):

    ax.plot(
        gdf["Phase"],
        gdf["mean"],
        marker="o",
        linewidth=2,
        label=group,
        color=GROUP_COLORS.get(group, "black")
    )

    ax.fill_between(
        gdf["Phase"],
        gdf["mean"] - 1.96 * gdf["sem"],
        gdf["mean"] + 1.96 * gdf["sem"],
        alpha=0.18,
        color=GROUP_COLORS.get(group, "black")
    )

ax.set_xlabel("Phase")
ax.set_ylabel("Alpha-band wPLI connectivity")
ax.set_title("Phase-resolved alpha-band connectivity")

# Legend outside plot
ax.legend(
    title="Group",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False
)

ax.spines[['top', 'right']].set_visible(False)

# tighten y-range without exaggeration
ax.set_ylim(0.18, 0.50)

plt.tight_layout(rect=[0, 0, 0.85, 1])

plt.savefig(
    os.path.join(FIG_DIR, "Fig2_phase_group_clean.png"),
    bbox_inches="tight"
)
plt.savefig(
    os.path.join(FIG_DIR, "Fig2_phase_group_clean.pdf"),
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Figure 3 — Network distribution (raincloud-style)
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

sns.violinplot(
    data=model_df,
    x="Network",
    y="Connectivity",
    inner=None,
    color="lightgray",
    linewidth=1,
    ax=ax
)

sns.stripplot(
    data=model_df,
    x="Network",
    y="Connectivity",
    color="black",
    size=2.5,
    alpha=0.25,
    jitter=0.22,
    ax=ax
)

sns.pointplot(
    data=model_df,
    x="Network",
    y="Connectivity",
    errorbar="se",
    color="red",
    join=False,
    capsize=0.1,
    ax=ax
)

ax.set_xlabel("Network")
ax.set_ylabel("Alpha-band wPLI connectivity")
ax.set_title("Network-level alpha-band connectivity")

ax.spines[['top', 'right']].set_visible(False)

plt.xticks(rotation=30)

# tighten axis slightly
ax.set_ylim(0.18, 0.50)

plt.tight_layout()

plt.savefig(
    os.path.join(FIG_DIR, "Fig3_network_raincloud.png"),
    bbox_inches="tight"
)
plt.savefig(
    os.path.join(FIG_DIR, "Fig3_network_raincloud.pdf"),
    bbox_inches="tight"
)

plt.close()