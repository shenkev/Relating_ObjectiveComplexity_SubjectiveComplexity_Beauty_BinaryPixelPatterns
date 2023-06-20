#!/usr/bin/env python
# coding: utf-8

# # Descriptive Analysis

"""
Author: Surabhi S Nath
Description: This script implements descriptive analysis on the data.
"""

import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', 10000)
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_theme()
sns.set(font_scale=1.8, font="serif", rc={'axes.facecolor':'white'})
from scipy import stats


def pearsonr_ci(x, y, alpha=0.05):
    """ 
    Calculate Pearson correlation along with the confidence interval using scipy and numpy
    
    Parameters:
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    
    Returns:
      r : float
        Pearson's correlation coefficient
      pval : float
        The corresponding p value
      lo, hi : float
        The lower and upper bound of confidence intervals
      df: integer
        Degrees of freedom
    """
    
    assert len(x) == len(y)

    df = len(x) - 2
    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi, df


# Define variables
path = "../data/"
patterns_path = "../"
num_patterns = 216
num_trials = 60

# Load files
pattern_to_num = pickle.load(open("utils/pattern_to_num.pk", 'rb'))
num_to_pattern = {v: k for k, v in pattern_to_num.items()}
info = pd.read_csv(path + "info.csv")
num_rows = info.shape[0]
pattern_stats = pd.read_csv("utils/pattern_stats.csv")
stat_analysis_csv = pd.read_csv("utils/stat_analysis.csv")


# ## Filter Participants


# Remove subjects who failed all attention checks
all_attention_failed = list(info[info["num_failed_attention_checks"] == info["num_attention_checks"]]["subject"])
num_subjects = len(info[~info["subject"].isin(all_attention_failed)])
subjects = info[~info["subject"].isin(all_attention_failed)]["subject"].tolist()
stat_analysis_csv = stat_analysis_csv[~stat_analysis_csv["subject"].isin(all_attention_failed)]


# Read data and save to arrays
pattern_complexity_ratings = [[] for i in range(num_patterns)]
pattern_beauty_ratings = [[] for i in range(num_patterns)]
subject_complexity_responses = []
subject_beauty_responses = []
subject_patterns = []

# Save repeats info - first, second value separately for correlations, SD values, differences

# first, second value separately
firstrepeat_complexity = []
secondrepeat_complexity = []
firstrepeat_beauty = []
secondrepeat_beauty = []

# SD values
participant_repeats_complexity_sd = []
participant_repeats_beauty_sd = []

# differences
complexity_diff_in_repeats = []
beauty_diff_in_repeats = []
repeats_distances = []

for i, file in enumerate(np.sort(os.listdir(path))):
    if file[0:4] != "data":
        continue
    if file[5:13] in all_attention_failed:
        continue
    
    csv = pd.read_csv(path + file)

    # Repeats
    repeats = csv[csv["is_repeated"] == True]

    complexity_repeats = repeats.sort_values(by='pattern')["complexity_response"].to_numpy()
    beauty_repeats = repeats.sort_values(by='pattern')["beauty_response"].to_numpy()

    # store first rating and second rating separately
    firstrepeat_complexity.append(complexity_repeats[::2])
    secondrepeat_complexity.append(complexity_repeats[1::2])
    firstrepeat_beauty.append(beauty_repeats[::2])
    secondrepeat_beauty.append(beauty_repeats[1::2])

    # calculate mean standard deviation of repeats
    participant_repeats_complexity_sd.append(np.mean(repeats.groupby('pattern').std()['complexity_response'].values))
    participant_repeats_beauty_sd.append(np.mean(repeats.groupby('pattern').std()['beauty_response'].values))

    # Calculate difference in repeats
    repeats_diff = repeats.sort_values(by='pattern').diff()
    repeats_diff = repeats_diff[repeats_diff['pattern'] == 0]
    complexity_diff_in_repeats.append(repeats_diff['complexity_response'].abs().to_numpy())
    beauty_diff_in_repeats.append(repeats_diff['beauty_response'].abs().to_numpy())
    repeats_distances.append(repeats_diff['trial'].abs().to_numpy())


    # Record responses and pattern list
    beauty_responses = csv["beauty_response"].to_numpy()
    subject_beauty_responses.append(beauty_responses)
    complexity_responses = csv["complexity_response"].to_numpy()
    subject_complexity_responses.append(complexity_responses)
    patterns = csv["pattern"].to_numpy()
    subject_patterns.append(patterns)
    
    # Record responses per pattern
    for i,p in enumerate(patterns):
        pattern_complexity_ratings[p-1].append(complexity_responses[i])
        pattern_beauty_ratings[p-1].append(beauty_responses[i])


# ## Correlations - Figure 4, Figure AIII.1

df_corr = stat_analysis_csv[["complexity_rating", "beauty_rating", "density", "entropy", "mean_entropy", "entropy_of_means", "LSC", "KC", "intricacy_4", "intricacy_8", "local_asymm",	"Hasymm",	"Vasymm", "quadtree", "num_active_rules"]]

# Uncomment to print significances, CI:
'''
for c1 in df_corr.columns:
    for c2 in df_corr.columns:
        print(c1, "\t", c2)
        corr, sig, cil, cih, df = pearsonr_ci(df_corr[c1].values, df_corr[c2].values)
        print(np.round(corr,2), "\t", np.round(sig, 3), "\t", np.round(cil, 2), "\t", np.round(cih, 2))
'''

# Rename columns for figures
df_corr = df_corr.rename(columns={  
                                "complexity_rating": "complexity rating",
                                "beauty_rating": "beauty rating",
                                "mean_entropy": "mean entropy",
                                "entropy_of_means": "entropy of means",
                                "intricacy_4": "intricacy",
                                "local_asymm": "local asymmetry",
                                "Hasymm": "horizontal asymmetry",
                                "Vasymm": "vertical asymmetry",
                                "num_active_rules": "\u03BB"
                                })

# Evaluate Pearson correlations
df_corr = df_corr.corr(method='pearson').round(decimals=2)

# Figure 4
df_corr_measures = df_corr.drop(['complexity rating', 'beauty rating', 'intricacy_8', 'quadtree'], axis=1).drop(['complexity rating', 'beauty rating', 'intricacy_8', 'quadtree'], axis=0)
plt.figure(figsize=(16,10))
matrix = np.triu(df_corr_measures)
ax = sns.heatmap(df_corr_measures, annot=True, fmt=".2f", cmap="hot_r", mask=matrix)
plt.tight_layout()
plt.savefig("plots/Figure_4_correlations.png")

# Figure AIII.1
df_corr_ratings = df_corr[["complexity rating", "beauty rating"]]
plt.figure(figsize=(12,7))
ax = sns.heatmap(df_corr_ratings, annot=True, fmt=".2f", cmap="hot_r")
plt.tight_layout()
plt.savefig("plots/Figure_AIII_1_correlations.png")


# ## Participant Qualitative Open-ended answers

# Collect open-ended responses for the 4 questions
open_ended_answers = pd.DataFrame({"Complexity Startegy": info["complexity_strategy"].tolist(),
                                  "Beauty Strategy": info["beauty_strategy"].tolist(), 
                                  "Most Complex": info["most_complex"].tolist(), 
                                  "Most beautiful": info["most_beautiful"].tolist()})
open_ended_answers


# ## Average Ratings Analysis - Figure AIII.5, AIII.6

# ### Average Beauty vs Average Complexity Plot - Figure AIII.5

# get patternwise mean and variance
stat_analysis_csv_patternwisemean = stat_analysis_csv.groupby("pattern").mean()
stat_analysis_csv_patternwisevar = stat_analysis_csv.groupby("pattern").var()

# assign entropy bin based on entropy of means
bins_entropy = [0, np.sort(stat_analysis_csv_patternwisemean["entropy_of_means"])[num_patterns//3], np.sort(stat_analysis_csv_patternwisemean["entropy_of_means"])[(num_patterns//3) * 2], np.inf]
names_entropy = ['low', 'medium', 'high']
stat_analysis_csv_patternwisemean["entropy_of_means_bin"] = pd.cut(stat_analysis_csv_patternwisemean["entropy_of_means"], bins_entropy, labels=names_entropy)

# assign symmetry bin based on Hasymm and Vasymm
def get_symm_bin(row):
    if row["Hasymm"] + row["Vasymm"] == 0:
        return "full symmetry"
    if row["Hasymm"] * row["Vasymm"] == 0:
        return "unidirection symmetry"
    return "partial/full asymmetry"
stat_analysis_csv_patternwisemean["symmetry_bin"] = stat_analysis_csv_patternwisemean.apply(lambda row: get_symm_bin(row), axis=1)


# Figure AIII.5(A)
sns.scatterplot(data=stat_analysis_csv_patternwisemean, x='complexity_rating', y='beauty_rating', hue='symmetry_bin')
plt.xlabel("Average Complexity Ratings")
plt.ylabel("Average Beauty Ratings")
sns.set(font_scale=1.5, font="serif")
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,1,0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# Figure AIII.5(B)
sns.scatterplot(data=stat_analysis_csv_patternwisemean, x='complexity_rating', y='beauty_rating', hue='entropy_of_means_bin')
plt.xlabel("Average Complexity Ratings")
plt.ylabel("Average Beauty Ratings")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# ### GMM Analysis on Average ratings - Figure AIII.6

# Fit GMM on average complexity ratings vs average beauty ratings coloured by symmetry

from sklearn.model_selection import train_test_split
import gmr
from gmr import GMM

X = stat_analysis_csv_patternwisemean["complexity_rating"].values.reshape(-1, 1)
X_test = X[:40,:]
X_train = X[40:,:]
y = stat_analysis_csv_patternwisemean["beauty_rating"].values.reshape(-1, 1)
y_test = y[:40,:]
y_train = y[40:,:]

data = np.concatenate([X_train, y_train], 1)

# 2 components for the bimodality
gmm = GMM(n_components=2, random_state=0)
gmm.from_samples(data)

# predictions on train data
data = np.linspace(np.min(y_train), np.max(y_train), 1000).reshape(-1, 1)
preds_train = gmm.predict([1], data)

# predictions on test data - not used
preds_test = gmm.predict([1], y_test)
RMSE_test = np.sqrt(np.sum((preds_test - X_test)**2)/len(preds_test))
# print(RMSE_test)

# Figure AIII.6
sns.scatterplot(data=stat_analysis_csv_patternwisemean, x='complexity_rating', y='beauty_rating', hue='symmetry_bin')
plt.plot(preds_train, data, c = "black", linestyle='dashed')
plt.xlabel("Average Complexity Ratings")
plt.ylabel("Average Beauty Ratings")
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,1,0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# ## Consistency of Repeats - Figures AIII.3 and AIII.4

# ### Mean Standard Deviation of Repeats - Figure AIII.3

# Figure AIII.3
# Plot histogram of participant mean standard deviation of repeats

fig, ax = plt.subplots(1, 2, figsize=(20,5))

ax[0].hist(participant_repeats_complexity_sd)
ax[0].set_xlabel("\nAverage Standard Deviation of Repeats")
ax[0].set_ylabel("Number of Participants\n")
ax[0].set_title("Complexity Ratings")

ax[1].hist(participant_repeats_beauty_sd)
ax[1].set_xlabel("\nAverage Standard Deviation of Repeats")
ax[1].set_ylabel("Number of Participants\n")
ax[1].set_title("Beauty Ratings")


# ### Test-retest Correlations - Figure AIII.4

# Figure AIII.4
# plot histogram of participant test-retest correlations

firstrepeat_complexity = np.array(firstrepeat_complexity)
secondrepeat_complexity = np.array(secondrepeat_complexity)
firstrepeat_beauty = np.array(firstrepeat_beauty)
secondrepeat_beauty = np.array(secondrepeat_beauty)

avg_ppt_corr_complexity = []
for pair in zip(firstrepeat_complexity, secondrepeat_complexity):
    avg_ppt_corr_complexity.append(np.corrcoef(pair[0], pair[1])[0,1])

avg_ppt_corr_beauty = []
for pair in zip(firstrepeat_beauty, secondrepeat_beauty):
    avg_ppt_corr_beauty.append(np.corrcoef(pair[0], pair[1])[0,1])

fig, ax = plt.subplots(1, 2, figsize=(20,5))

ax[0].hist(avg_ppt_corr_complexity)
ax[0].set_xlabel("\nTest-retest Correlation")
ax[0].set_ylabel("Number of Participants\n")
ax[0].set_title("Complexity Ratings")

ax[1].hist(avg_ppt_corr_beauty)
ax[1].set_xlabel("\nTest-retest Correlation")
ax[1].set_ylabel("Number of Participants\n")
ax[1].set_title("Beauty Ratings")


# ## Max Discrepancy - Figure 8

# Figure 8
# Display patterns that have max discrepancy in the onkective complexity measure and mean subjective complexity ratings 
max_discrepancy_pattnnums = [195, 66, 92, 180, 81, 23, 203, 190, 204, 207] # list of pattern numbers from MixedEffectsModelling.R
max_discrepancy_patterns = np.vectorize(num_to_pattern.get)(max_discrepancy_pattnnums) # get corresponding patterns

fig, ax = plt.subplots(2, 5, figsize=(20,5))
for ind, pattern in enumerate(max_discrepancy_patterns):
    image = plt.imread(patterns_path + pattern)
    ax[ind//5,ind%5].imshow(image)
    ax[ind//5,ind%5].axis('off')


# ## Pattern Visualisation - Figure 6

# ### Average response - Figure 6A-D

to_display = 5


avg_pattern_complexity_ratings = stat_analysis_csv_patternwisemean["complexity_rating"].to_numpy()
avg_pattern_beauty_ratings = stat_analysis_csv_patternwisemean["beauty_rating"].to_numpy()


high_complexities = avg_pattern_complexity_ratings[np.argsort(avg_pattern_complexity_ratings)[-to_display:]] # get the highest average complexity ratings
high_complex_patternnums = np.argsort(avg_pattern_complexity_ratings)[-to_display:] + 1 # get the patternnums with highest average complexity ratings (convert numpy indexing to pattern num with + 1)
high_complex_patterns = np.vectorize(num_to_pattern.get)(high_complex_patternnums) # get corresponding pattern names

low_complexities = avg_pattern_complexity_ratings[np.argsort(avg_pattern_complexity_ratings)[:to_display]] # get the lowest average complexity ratings
low_complex_patternnums = np.argsort(avg_pattern_complexity_ratings)[:to_display] + 1 # get the patternnums with lowest average complexity ratings (convert numpy indexing to pattern num with + 1)
low_complex_patterns = np.vectorize(num_to_pattern.get)(low_complex_patternnums) # get corresponding pattern names

high_beauties = avg_pattern_beauty_ratings[np.argsort(avg_pattern_beauty_ratings)[-to_display:]] # get the highest average beauty ratings
high_beauty_patternnums = np.argsort(avg_pattern_beauty_ratings)[-to_display:] + 1 # get the patternnums with highest average beauty ratings (convert numpy indexing to pattern num with + 1)
high_beauty_patterns = np.vectorize(num_to_pattern.get)(high_beauty_patternnums) # get corresponding pattern names

low_beauties = avg_pattern_beauty_ratings[np.argsort(avg_pattern_beauty_ratings)[:to_display]] # get the lowest average beauty ratings 
low_beauty_patternnums = np.argsort(avg_pattern_beauty_ratings)[:to_display] + 1 # get the patternnums with lowest average beauty ratings (convert numpy indexing to pattern num with + 1)
low_beauty_patterns = np.vectorize(num_to_pattern.get)(low_beauty_patternnums) # get corresponding pattern names


# #### High Complexity - Figure 6A

# Figure 6A
fig, ax = plt.subplots(1,5,figsize=(20,5))
for i in range(to_display):
    image = plt.imread(patterns_path + high_complex_patterns[i])
    ax[i].imshow(image)
    ax[i].set_title(np.round(high_complexities[i],2))
    ax[i].axis('off')


# #### Low Complexity - Figure 6B

# Figure 6B
fig, ax = plt.subplots(1,5,figsize=(20,5))
for i in range(to_display):
    image = plt.imread(patterns_path + low_complex_patterns[i])
    ax[i].imshow(image)
    ax[i].set_title(np.round(low_complexities[i],2))
    ax[i].axis('off')


# #### High Beauty - Figure 6C

# Figure 6C
fig, ax = plt.subplots(1,5,figsize=(20,5))
for i in range(to_display):
    image = plt.imread(patterns_path + high_beauty_patterns[i])
    ax[i].imshow(image)
    ax[i].set_title(np.round(high_beauties[i],2))
    ax[i].axis('off')


# #### Low Beauty - Figure 6D

# Figure 6D
fig, ax = plt.subplots(1,5,figsize=(20,5))
for i in range(to_display):
    image = plt.imread(patterns_path + low_beauty_patterns[i])
    ax[i].imshow(image)
    ax[i].set_title(np.round(low_beauties[i],2))
    ax[i].axis('off')


# ### Variance in Responses - Figure 6E-H

var_pattern_complexity_ratings = stat_analysis_csv_patternwisevar["complexity_rating"].to_numpy()
var_pattern_beauty_ratings = stat_analysis_csv_patternwisevar["beauty_rating"].to_numpy()


confusing_complexities = var_pattern_complexity_ratings[np.argsort(var_pattern_complexity_ratings)[-to_display:]] # get the highest variance complexity ratings
confusing_patternnums_complexity = np.argsort(var_pattern_complexity_ratings)[-to_display:] + 1 # get the patternnums with highest variance complexity ratings (convert numpy indexing to pattern num with + 1)
confusing_patterns_complexity = np.vectorize(num_to_pattern.get)(confusing_patternnums_complexity) # get corresponding pattern names

consistent_complexities = var_pattern_complexity_ratings[np.argsort(var_pattern_complexity_ratings)[:to_display]] # get the lowest variance complexity ratings
consistent_patternnums_complexity = np.argsort(var_pattern_complexity_ratings)[:to_display] + 1 # get the patternnums with lowest variance complexity ratings (convert numpy indexing to pattern num with + 1)
consistent_patterns_complexity = np.vectorize(num_to_pattern.get)(consistent_patternnums_complexity) # get corresponding pattern names

confusing_beauties = var_pattern_beauty_ratings[np.argsort(var_pattern_beauty_ratings)[-to_display:]] # get the highest variance beauty ratings
confusing_patternnums_beauty = np.argsort(var_pattern_beauty_ratings)[-to_display:] + 1 # get the patternnums with highest variance beauty ratings (convert numpy indexing to pattern num with + 1)
confusing_patterns_beauty = np.vectorize(num_to_pattern.get)(confusing_patternnums_beauty) # get corresponding pattern names

consistent_beauties = var_pattern_beauty_ratings[np.argsort(var_pattern_beauty_ratings)[:to_display]] # get the lowest variance beauty ratings 
consistent_patternnums_beauty = np.argsort(var_pattern_beauty_ratings)[:to_display] + 1 # get the patternnums with lowest variance beauty ratings (convert numpy indexing to pattern num with + 1)
consistent_patterns_beauty = np.vectorize(num_to_pattern.get)(consistent_patternnums_beauty) # get corresponding pattern names


# #### Consistent Patterns Complexity - Figure 6E

# Figure 6E
fig, ax = plt.subplots(1,5,figsize=(20,5))
for i in range(to_display):
    image = plt.imread(patterns_path + consistent_patterns_complexity[i])
    ax[i].imshow(image)
    ax[i].set_title(np.round(consistent_complexities[i],2))
    ax[i].axis('off')


# #### Confusing Patterns Complexity - Figure 6F

# Figure 6F
fig, ax = plt.subplots(1,5,figsize=(20,5))
for i in range(to_display):
    image = plt.imread(patterns_path + confusing_patterns_complexity[i])
    ax[i].imshow(image)
    ax[i].set_title(np.round(confusing_complexities[i],2))
    ax[i].axis('off')


# #### Consistent Patterns Beauty - Figure 6G

# Figure 6G
fig, ax = plt.subplots(1,5,figsize=(20,5))
for i in range(to_display):
    image = plt.imread(patterns_path + consistent_patterns_beauty[i])
    ax[i].imshow(image)
    ax[i].set_title(np.round(consistent_beauties[i],2))
    ax[i].axis('off')


# #### Confusing Patterns Beauty - Figure 6H

# Figure 6H
fig, ax = plt.subplots(1,5,figsize=(20,5))
for i in range(to_display):
    image = plt.imread(patterns_path + confusing_patterns_beauty[i])
    ax[i].imshow(image)
    ax[i].set_title(np.round(confusing_beauties[i],2))
    ax[i].axis('off')