import numpy as np
import pandas as pd
import os
from datetime import datetime

from src.preprocessing import (get_outliers_removed, 
                              get_maximum_hold_times,
                              get_sample_hold_result,
                              get_imputed_table,
                              action_statistics,
                              correct_retrieved_table)

# to be moved to yaml file in future
dtypes = {"icustay_id":np.float64,
"subject_id":np.float64,
"hadm_id":np.float64,
"start_time":str,
"first_admit_age":np.float64,
"gender":str,
"weight":np.float64,
"icu_readm":np.float64,
"elixhauser_score":np.float64,
"sofa":np.float64,
"sirs":np.float64,
"gcs":np.float64,
"heartrate":np.float64,
"sysbp":np.float64,
"diasbp":np.float64,
"meanbp":np.float64,
"shockindex":np.float64,
"resprate":np.float64,
"resprate_spont":np.float64,
"resprate_high":np.float64,
"resprate_set":np.float64,
"tempc":np.float64,
"spo2":np.float64,
"potassium":np.float64,
"sodium":np.float64,
"chloride":np.float64,
"glucose":np.float64,
"bun":np.float64,
"creatinine":np.float64,
"magnesium":np.float64,
"calcium":np.float64,
"ionizedcalcium":np.float64,
"carbondioxide":np.float64,
"sgot":np.float64,
"sgpt":np.float64,
"bilirubin":np.float64,
"albumin":np.float64,
"hemoglobin":np.float64,
"wbc":np.float64,
"platelet":np.float64,
"ptt":np.float64,
"pt":np.float64,
"inr":np.float64,
"ph":np.float64,
"pao2":np.float64,
"paco2":np.float64,
"base_excess":np.float64,
"bicarbonate":np.float64,
"lactate":np.float64,
"pao2fio2ratio":np.float64,
"mechvent":np.float64,
"fio2":np.float64,
"urineoutput":np.float64,
"vaso_total":np.float64,
"iv_total":np.float64,
"cum_fluid_balance":np.float64,
"peep":np.float64,
"tidal_volume":np.float64,
"tidal_volume_obs":np.float64,
"tidal_volume_set":np.float64,
"tidal_volume_spont":np.float64,
"plateau_pressure":np.float64,
"peak_pressure":np.float64,
"hospmort90day":np.float64,
"dischtime":str,
"deathtime":str}

# to be moved to yaml file in future
parse_dates = ['start_time', 'deathtime', 'dischtime']

# path to be moved to yaml file in future
d_in = pd.read_csv('./data_in/cohort_500_patients.csv',
                   parse_dates=parse_dates,
                   dtype=dtypes) # read the patient files
#d_in = d_in[d_in['icustay_id'] == 200009.00000]
save_tables = {}
_version = 7.3
exec_start = datetime.now().strftime('%Y-%m-%d_%H-%M')

print('####  FIND MECHANICALLY VENTILATED ONES  ####')
counts=pd.DataFrame(index=[0])
icustay_id_groups = d_in.groupby('icustay_id') #group by thier id
stats = icustay_id_groups[['fio2', 'peep', 'tidal_volume']].mean() #select these 3 columns and get their mean

counts = (~stats.isnull()).sum() #return to false all r/c after true
counts.index = [f'count_{c}' for c in counts.index]
print (counts.index)
counts['count_fio2peepTV'] = (~stats.isnull()).product(axis=1).sum() #return to false all r/c after true, add new cloumn and add "1" to it
for c in stats.columns:
    print(c)
stats = stats.rename(columns={c:f'mean_{c}' for c in stats.columns})

# Find mechanically ventilated ones with ideal body weight
IBW = pd.read_csv('./data_in/idealbodyweight.csv')
#IBW = IBW[IBW['icustay_id'] == 200009.00000]
ibw_over_0 = IBW.ideal_body_weight_kg>0 # weights more than 0
icustay_non_null = ~IBW.icustay_id.isna()
IBW = IBW[ibw_over_0&icustay_non_null][['icustay_id', 'ideal_body_weight_kg']] #adding data from csv with weights

df = d_in.merge(IBW, how='left', on='icustay_id') #adding weights to the dataframe
df['TVnormalized'] = (df['tidal_volume']/df['ideal_body_weight_kg']).astype(np.float64)

icustay_id_groups = df.groupby('icustay_id')
stats['mean_TVnormalized'] = icustay_id_groups.TVnormalized.mean()
counts['count_fio2peepTVnormalized'] = (~stats[['mean_fio2', 'mean_peep', 'mean_TVnormalized']].isnull()).product(axis=1).sum() #doubt

icus_with_fio2peepTV = (~stats[['mean_fio2', 'mean_peep', 'mean_tidal_volume']].isnull()).product(axis=1)==1
icus_with_fio2peepTVnormalized = (~stats[['mean_fio2', 'mean_peep', 'mean_TVnormalized']].isnull()).product(axis=1)==1

icus_with_fio2peepTVnormalized_set = set(stats[icus_with_fio2peepTVnormalized].index)

data = df[df.icustay_id.isin(icus_with_fio2peepTVnormalized_set)]
#data = df[df.icustay_id.isin(set(stats[icus_with_fio2peepTV].index))]
print(f'Data size reduction {df.shape}->{data.shape} ; ({df.shape[0]-data.shape[0]} rows less)')

print('####  FIND AND REMOVE OUTLIERS  ####')
data_raw = data.copy()
save_tables['data_raw'] = data_raw

# Discard the variables that are not supposed to have outliers. These can
# contain id's, binary variables (like mechanical ventilation), scores that
# are limited by nature (like SOFA) and dates.

# to be moved to yaml file in future
none_outliers=['icustay_id','subject_id','hadm_id','start_time',
    'gender','icu_readm','elixhauser_score','sofa','sirs','gcs',
    'mechvent','hospmort90day','dischtime','deathtime']
var_names = list(data.columns.difference(none_outliers))

data.boxplot(column='heartrate', figsize=(13,10), fontsize=16)

# Remove the outliers
data = get_outliers_removed(data, var_names)

# Store those nonimputed drug values for later so that no undocumented drug
# instance will occur in data.
data_med = data.loc[:, ['vaso_total', 'iv_total']]
data_med = data_med.fillna(0)
save_tables['data_med'] = data_med

# Remove also manually some values less than a threshold.
peep_cond = data.peep<0.05
TVnorm_cond = (data.TVnormalized<0.1)|(data.TVnormalized>25) #normialize the data
data.loc[:, 'peep'] = np.where(peep_cond, None, data.peep).astype(np.float64)
data.loc[:, 'TVnormalized'] = np.where(TVnorm_cond, None, data.TVnormalized).astype(np.float64)

# Fill missing values as long as they are in duration of a hold time
print('####  FILL MISSING VALUES  ####')
data_outliersremoved = data.copy();
save_tables['data_outliersremoved'] = data_outliersremoved

# to be moved to yaml file in future
not_holdtime=['icustay_id','subject_id','hadm_id','start_time',
    'gender','icu_readm','elixhauser_score','sofa','sirs','gcs','sgot','sgpt',
    'mechvent','hospmort90day','pao2fio2ratio','dischtime','deathtime']
var_names = list(data.columns.difference(not_holdtime))

hold = get_maximum_hold_times(data, var_names, 4, 0.9);
print('##  AQUIRING HOLD TIMES  ##')
print(hold)

# takes ~5.5 minutes on my PC with ~50k patients; takes 3 sec for 500 patients; I don't know on Agatha computer in lab
data.loc[:, var_names] = data.groupby('icustay_id')[['start_time']+var_names].apply(get_sample_hold_result, 
                                                           hold=hold.astype('timedelta64[h]').to_dict(), 
                                                           var_names=var_names)

#Calculate PaO2/Fio2 again after sample and hold. 
data.pao2fio2ratio=100*data.pao2/data.fio2; 

#  Impute Remaining Missing Values
print('####  IMPUTE REMAINING MISSING VALUES  ####')
data_sampleholded = data.copy();
save_tables['data_sampleholded'] = data_sampleholded

print('# Saving intermediate results achieved so far. #')
path_intermediate = os.path.join(os.getcwd(), f'data_out/v.{_version}/intermediate/{exec_start}')
os.makedirs(path_intermediate, exist_ok=True)
for name in save_tables:
    path = os.path.join(path_intermediate, '.'.join([name, 'pkl']))
    save_tables[name].to_pickle(path)
print('Done.')

## imputation step
print('# Proper imputation step #')
# to be moved to yaml file in future
not_impute = ['icustay_id','subject_id','hadm_id','start_time',
    'gender','sgot','sgpt', 'hospmort90day','dischtime','deathtime']
var_names = list(data.columns.difference(not_impute))

data_array = data.loc[:, var_names].to_numpy()

# Gender is a string info but can be used for imputation as a binary variable.
data_array = np.column_stack([data_array, data.loc[:,'gender']=='M']) #Doubt

#~3min 30s for 500 patients data on personal PC
K=10; blockSize=400; rows_impute=data_array.shape[0] #Doubt
data_array=get_imputed_table(data_array, K, blockSize, rows_impute)

# There were some data that were all NaN s at the beginning, the imputation
# algorithm cannot find a value for those so discard them
nan_rows = np.isnan(data_array).any(axis=1)
data_array = data_array[~nan_rows]

data = pd.DataFrame(data_array[:, 0:-1], columns=var_names)

# gcs must be an integer and mechvent must be binary
data.gcs = data.gcs.round(0).astype(int)
data.mechvent = np.where(data.mechvent>0.5, 1, 0)

# to be moved to yaml file in future
raw_cols = ['icustay_id','subject_id','hadm_id','start_time', 'gender','hospmort90day','dischtime','deathtime']


patient_info = data_raw.loc[~nan_rows, raw_cols]
data = pd.concat([patient_info[raw_cols[:5]], data, patient_info[raw_cols[5:]]], axis=1)

print('####  ELIMINATE INSUFFICENT DATAPOINTS  ####')
data_filtered = action_statistics(data, data_outliersremoved, data_raw, data_sampleholded)

print('####  CONVERT TABLE  ####')
MIMIC_table = correct_retrieved_table(data_filtered)

print('####  SAVE FINAL RESULTS  ####')
path_intermediate = os.path.join(os.getcwd(), f'data_out/v.{_version}/after_preprocessing/{exec_start}')
os.makedirs(path_intermediate, exist_ok=True)
path = os.path.join(path_intermediate, '.'.join(['MIMIC_table', 'pkl']))
MIMIC_table.to_pickle(path)
MIMIC_table.to_csv(path.replace('pkl', 'csv'))
print('####  DONE  ####')