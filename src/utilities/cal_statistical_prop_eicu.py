import pandas as pd
import os


cwd=os.path.dirname(os.path.abspath(__file__))

# input and output files loaded:
output_data_path = os.path.join(cwd,"..",'..', "data","lab_values", "eICU_lab_results.csv")
input_data_path = os.path.join(cwd,"..",'..', "data","lab_values", "eICU_edited.csv")
statistical_df_data_path = os.path.join(cwd,"..",'..',"data", "lab_values", "eicu_statistical.csv")

def read_data(window_length):
    # read the input and output files
    output_data = pd.read_csv(output_data_path, )

    # filter the dataset to include the icustays with more than window_length timestamps (window_length* 4 hours )
    output_data = output_data.groupby('icustayid').filter(lambda x: len(x) > window_length)


    #part to generate zeros and ones count for the lab results
    output_data_cp=output_data.copy()
    print("no of sequences ", output_data.shape[0])
    output_data_cp.loc['sum_ones']=pd.Series(output_data_cp.sum(axis=0))

    output_data_cp.loc['sum_zeros'] = 388571-output_data_cp.loc['sum_ones']
    output_data_cp=output_data_cp.tail(2)
    balance_analysis=output_data_cp.iloc[:,2:]
    print(balance_analysis)
    balance_analysis=balance_analysis.transpose()
    balance_analysis["max"]=balance_analysis.max(axis=1)
    balance_analysis["max"] = ((balance_analysis["max"]/388571)*100).astype(int)
    balance_analysis["balanced"] = balance_analysis["max"] < 70
    print(balance_analysis)

    balance_analysis.to_csv(os.path.join(cwd, "..", '..', "data", "lab_values", "lab_results_zeros_counts_eicu.csv"))

    lab_values = pd.read_csv(input_data_path)
    lab_values.drop("bloc", axis=1, inplace=True)
    lab_names = lab_values.columns.tolist()
    lab_names.remove("icustayid")

    grouped_mean = lab_values.groupby("icustayid").mean()
    grouped_max = lab_values.groupby("icustayid").max()
    grouped_min = lab_values.groupby("icustayid").min()
    grouped_std = lab_values.groupby("icustayid").std()

    means_mean = grouped_mean.mean()
    means_std = grouped_std.mean()
    means_max = grouped_max.max()
    means_min = grouped_min.min()
    statistical_df = pd.DataFrame(columns=["feature_name", "mean", "std", "max", "min"])
    statistical_df["feature_name"] = lab_names
    statistical_df["mean"] = means_mean.tolist()
    statistical_df["std"] = means_std.tolist()
    statistical_df["max"] = means_max.tolist()
    statistical_df["min"] = means_min.tolist()
    print(statistical_df)

    statistical_df.to_csv(statistical_df_data_path, index=False)



if __name__ == '__main__':
    read_data(window_length=5)


