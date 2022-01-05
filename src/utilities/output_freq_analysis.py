"""

* Copyright (C) Clinomic, GmbH - All Rights Reserved

* Unauthorized copying of this file, via any medium is strictly prohibited

* Proprietary and confidential

*

"""
"""
This program does frequency analysis on the patients output_lab_values_mat.csv dataset 
# it counts the number of ones in each patients vector and chooses the patients with ones sum less than a specific number
# save both the frequency analysis and the subset




"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(16, 9))

# paths for files to be read
path = Path(__file__).resolve().parents[2]
csv_file = os.path.join(path,"data", "lab_values", "lab_results.csv")
save_file_path = os.path.join(path,"data", "lab_values", "mimi_output_freq_analysis.csv")
fig_file_path = os.path.join(path,"data", "figures", "results_freq_analysis.png")
subpat_file_path = os.path.join(path,"data", "lab_values", "lab_results_freq.csv")


def read_data():
    data = pd.read_csv(csv_file)
    return data


def analyze_results():
    # read the lab results data
    csv = read_data()
    full_Data_length = csv.shape[0]
    print("original data shape: ", csv.shape)
    csv_ids = csv.loc[:, ["icustayid","bloc"]]
    csv_edited = csv.drop(["icustayid","bloc"], axis=1)
    # calculate the unique patients binary vectors
    (unique, counts) = np.unique(csv_edited.values, axis=0, return_counts=True)
    print(unique,counts)
    # calculate the number of ones in each binary vector of the original dataset
    csv_edited["sum"] = csv_edited.sum(axis=1)

    #csv_edited_full = pd.concat([csv_ids, csv_edited], axis=1)
    # the full dataset with the sum calculated
    print(csv_edited)
    # calculate the number of ones in each unique binary vector from the dataset
    unique_pd = pd.DataFrame(data=unique, columns=[f'f{i}' for i in range(25)])
    unique_pd["sum"] = unique_pd.sum(axis=1)
    unique_pd["counts"] = counts
    # sort the values decreasingly
    unique_pd = unique_pd.sort_values("counts", ascending=False).reset_index(drop=True)
    frequency = unique_pd.loc[:, ["sum", "counts"]]
    # group the unique vectors with the same ones sum
    grouped = frequency.groupby('sum')
    freq = grouped.sum()
    freq = freq.reset_index(drop=False)
    print("frequency ds: \n", freq)
    # calculate the cumulative count and the percentage

    cum_count_pd = pd.DataFrame()
    for index, rows in freq.iterrows():
        x = freq.loc[:index, "counts"].sum()
        cum_count_pd = cum_count_pd.append([x])
    values = cum_count_pd.iloc[:, 0].reset_index(drop=True)
    freq["cum_count"] = values
    freq["percentage"] = round((freq["cum_count"] / full_Data_length) * 100, 2)
    print(freq)
    csv_edited.loc[len(csv_edited)] = csv_edited.sum(axis=0)
    print(csv_edited)
    # choose patients with less than 15 ones in their vector
    #sub_patients = csv_edited_full.loc[csv_edited_full["sum"] < 22]
    #sub_patients = sub_patients.reset_index(drop=True)

    # plot the percentage with sum
    plt.bar(x=freq["sum"], height=freq["percentage"], log=True)
    plt.ylabel('Frequency')
    plt.draw()

    # save a csv file of the frequency analysis and the plot
    freq.to_csv(save_file_path, header=True, index=False)
    #sub_patients = sub_patients.drop(["sum"], axis=1)
    #print(sub_patients)
    # save the dataset with patients that has less than x number of ones in their vector
    #sub_patients.to_csv(subpat_file_path, header=True, index=False)
    plt.savefig(fig_file_path)
    plt.show()


if __name__ == "__main__":
    analyze_results()
