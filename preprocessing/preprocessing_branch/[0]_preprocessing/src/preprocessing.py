import numpy as np
from numba import jit, njit, prange
import pandas as pd
import datetime as dt
from datetime import datetime

def get_outliers_removed(data_in: pd.DataFrame, varnames: list) -> pd.DataFrame:
    """
    This function removes the outliers within the input data i.e. converts
    them to NaN s, using Tukey's method. It basically finds the 25th and
    75th percentiles of the input data. The difference between the 25th
    and 75th percentiles are taken (interquartile distance - IQR)
    and multiplied by a scalar (here 1.5).
    This last value is subtracted from 25th percentile and added to 75th
    percentile. Those are defining the limits for 'acceptable data' and
    anything outside of this interval is considered as an 'outlier'.
    
    data_in: MxN input pd.DataFrame which contains outliers.
    
    varnames: 1xN cell that contains the names of the variables that may 
    contain outliers. Variables like ID's or dates should be excluded.
    
    data_out: MxN output pd.DataFrame which outliers are removed. Except for the
    excluded outliers, this is identical to data_in.
    
    Copy input data.
    data_out=data_in;
    """
    data_out = data_in.copy()
    Q1 = data_out[varnames].quantile(0.25)
    Q3 = data_out[varnames].quantile(0.75)
    IQR_adjusted = (Q3-Q1)*1.5
    mask = ~((data_out[varnames] < (Q1-IQR_adjusted))|(data_out[varnames] > (Q3+IQR_adjusted)))
    data_out.loc[:, varnames] = data_out.loc[:, varnames].where(mask)
    return data_out


def param_max_hold(parameter: pd.Series, resol: int=4, 
                   threshold: float=0.9) -> float:
    """
    This function calculates maximum hold times that will be used in
    sample-and hold algorithm for a single variable. 
    The counts of consecutive measurement time differences are obtained 
    and when their cumulative sum exceeds a threshold, the first value 
    this occurs is taken as the hold time.
    
    data: 1xN pd.Series, with icustay_id & start_time set as index
    resol: Resolution of time differences in hours.
    threshold: The threshold to define the maximum hold time.
    """
    # Exclude the rows if there is no measurement for the parameter. They
    # are not helpful for finding the measurement frequency.
    param = parameter.dropna()
    #if param.notnull:
    # Get back icustay_id & start_time columns
    data_param = param.reset_index()
    # Difference between consecutive icustay_id
    diff_icustay = data_param.icustay_id.diff().fillna(0)
    # Difference between consecutive measurements
    st_times = data_param.start_time;
    diff_time = st_times.diff()
    # Exclude the time differences that do not belong to same icustay, they  
    # do not give useful info for measurement frequencies (when
    # diff_icustay is different from zero).
    diff_time = diff_time[diff_icustay==0]
    # binn diff hours
    diff_hours = diff_time.dt.total_seconds()/3600
    bins = np.arange(0, (diff_hours.max()), resol)
    binned = np.digitize(diff_hours, bins=bins)
    indices, counts = np.unique(binned, return_counts=True)
    ratio = (counts/counts.sum()).cumsum()
    # -1 to get left bin edge, like in matlab, instead of right
    index = (indices[ratio>threshold]-1)[0]
    if index<=0: # correct edge case
        index=0
    return bins[index]


def get_maximum_hold_times(data: pd.DataFrame, varnames, 
                           resol: int, threshold: float) -> pd.Series:
    """
    This function calculates maximum hold times that will be used in
    sample-and hold algorithm for all relevant variables.
    The counts of consecutive measurement time differences are obtained 
    and when their cumulative sum exceeds a threshold, 
    the first value this occurs is taken as the hold time.
    
    data: MxN pd.DataFrame
    varnames: Names of the variables that will be filled in the
              sample-hold part and a corresponding hold time 
              will be defined
    resol: Resolution of time differences in hours.
    threshold: The threshold to define the maximum hold time.
    
    hold: (M,) pd.Series, containing maximum hold times for variables.
    """
    hold = (data.set_index(['icustay_id', 'start_time']) # filtering columns as indices
          .loc[:, [*varnames]] # get just relevant parameters
          .apply(param_max_hold, args=(resol, threshold), axis=0)) # get max hold times
    return hold


def param_sample_hold(parameter: pd.Series, hold: pd.Series,
                      datetime: pd.Series) -> pd.Series:
    var_name = parameter.name
    test_vals = parameter.values
    # Find the parameter that are missing/nonmissing for parameter of
    # interest
    idx_missing = np.nonzero(np.isnan(test_vals))[0]
    idx_nonmissing = np.nonzero(~np.isnan(test_vals))[0]
    if idx_nonmissing.size == 0:
        return test_vals
    # Here the indexes of samples to be filled for missing values are
    # calculated.
    sz1, sz2= idx_nonmissing.shape[0], idx_missing.shape[0];

    a = np.tile(idx_missing, (sz1, 1))
    b = np.tile(idx_nonmissing.T, (sz2,1))
    c = np.max(np.cumsum((a.T-b>0), axis=1), axis=1)

    samples = idx_nonmissing[c-1]
    # Fill a missing value if the closest nonmissing value is within
    # a maximum hold time and if that nonmissing value is measured
    # BEFORE the missing one.
    if samples.size > 0:
        dt_diff = (datetime.values[idx_missing-1]-datetime.values[samples]).astype('timedelta64[h]')
        measured_before = (dt_diff>=np.timedelta64(0,'h'))
        within_hold = (dt_diff<=hold[var_name])#hold.at[var_name])
        missviable = idx_missing[measured_before&within_hold]
        sampleviable = samples[measured_before&within_hold]
        #test_param.iloc[missviable] = test_param.iloc[sampleviable].values
        test_vals[missviable] = test_vals[sampleviable]
    return test_vals


def get_sample_hold_result(d, hold, var_names):
    # get general datetime index
    dt = d.loc[:, 'start_time']
    # only subset of features
    res_id = d.loc[:, [*var_names]]
    res_id = res_id.apply(param_sample_hold, 
                          hold=hold,
                          datetime=dt,
                          axis=0)
    return res_id


@njit(cache=True)
def min_max_scaler(A):
    # minmax function that will limit the columns of A within the range [0,1]
    mini = np.empty(A.shape[1])
    maxi = np.empty(A.shape[1])
    for i, col in enumerate(A.T):
        mini[i] = np.nanmin(col)
        maxi[i] = np.nanmax(col)
    A_out=(A-mini)/(maxi-mini)
    return A_out


@njit(cache=True)
def dist_present(xi,xj, missing_values=None):
    """
    distance between present values of the vectors, this is the
    application of formula (1) from 'Improved methods for the imputation
    of missing data by nearest neighboure methods'(Gerhard Tutz, Shahla
    Ramzan)
    """
    q = 2
    xi_mod = np.zeros(xi.shape[0])
    xj_mod = np.zeros(xj.shape[0])
    m_ij = np.zeros(xi.shape[0])
    mij_sum = 0
    for i in range(xi.shape[0]):
        xinan, xjnan = np.isnan(xi[i]), np.isnan(xj[i])
        if xinan and xjnan:
            pass
        elif xinan:
            xj_mod[i] = xj[i]
        elif xjnan:
            xi_mod[i] = xi[i]
        else:
            m_ij[i]=1
            mij_sum=mij_sum+1
            xi_mod[i] = xi[i]
            xj_mod[i] = xj[i]
    distance = (1/mij_sum*np.sum(np.abs(xi_mod-xj_mod)**q*m_ij))**(1/q)
    return distance


@njit(cache=True, parallel=True)
def pairwise_distances_parallel(d1, d2):
    tmp_matrix = np.zeros((d1.shape[0], d2.shape[0]))
    for i in prange(d1.shape[0]):
        for j in range(d2.shape[0]):
            tmp_matrix[i, j] = (dist_present(d1[i], d2[j]))
    return tmp_matrix


@njit(cache=True, parallel=True)
def impute_columns(rows, cols, I, data_array, distances, K):
    impute_values = np.zeros((rows.shape[0], K))
    weights = np.zeros((rows.shape[0], K))
    for count in prange(cols.shape[0]):
        I_tmp = I[:, rows[count]]
        # remove values that are NaN, they are not good for imputing
        I_tmp = I_tmp[~np.isnan(data_array[I_tmp, cols[count]])]
        if I_tmp.size != 0:
            idx = I_tmp[:min(K, I_tmp.shape[0])]
            impute_values[count,:idx.shape[0]] = data_array[idx, cols[count]]
            weights[count,:idx.shape[0]] = 1/distances[idx, rows[count]]
    return weights, impute_values


def split_given_size(array, block_size):
    return np.split(array, np.arange(block_size, array.shape[0], block_size))


def get_imputed_table(data, K, block_size, rows_impute):
    """
    this is a modified version of MATLAB's knnimpute function. It is
    simplified according to needs of the project. The reason for the
    modifcaiton is that especially due to usage of pdist function, MATLAB
    built-in function cannot operate with comparatively large data so this
    function is modified to be used with only single row to be imputed with
    the rest of the data.
    """
    start_at = datetime.now()
    # make a copy of original data in order to ensure no missing value is filled
    # by using an imputed value but rather with a value that originally existed
    imputed = data.copy()
    eps = np.finfo('float64').eps
    
    # in order to make distance calculation meaningful the columns should be in
    # the same range (here [0,1] range), otherwise the parameters with
    # inherently larger values may dominate distance calculation
    data_dist = min_max_scaler(data)
    
    print(f'{datetime.now():%H:%M:%S} :: Starting imputation process...')
    for chunk_num, chunk in enumerate(split_given_size(data_dist[:rows_impute], block_size)):
        print(f'{datetime.now():%H:%M:%S} :: Rows {chunk_num*block_size} - {(chunk_num+1)*block_size-1} starting.')
        distances = pairwise_distances_parallel(data_dist, chunk)

        # there are many 0 distances , this will at the end cause corresponding 
        # weights to be NaN but instead those close values should be favored, 
        # for that reason they are equated to be a really small value (i.e. eps)
        distances[distances==0] = eps

        I = np.argsort(distances, axis=0)
        I = I[1:]
        nans = np.argwhere(np.isnan(chunk))
        nans = nans[nans[:,1].argsort()]
        rows, cols = nans[:, 0], nans[:, 1]

        impute_values = np.zeros((rows.shape[0], K))
        weights = np.zeros((rows.shape[0], K))

        print(f'\t{datetime.now():%H:%M:%S} :: Rows {chunk_num*block_size} - {(chunk_num+1)*block_size-1} inner loop.')
        weights, impute_values = impute_columns(rows, cols, I, data, distances, K)

        # some weights can be infinity which will turn a NaN result for imputing value
        weights[~np.isfinite(weights)]=0

        weights_normalized = (weights.T/weights.sum(axis=1)).T
        imputed[rows+(chunk_num*block_size), cols] = (weights_normalized*impute_values).sum(axis=1)
    print(f'DONE. Took {datetime.now()-start_at} in total')
    return imputed

#IN CASE JIT COMPUATION FAILS....
# def dist_present(xi,xj):
#     """
#     distance between present values of the vectors, this is the
#     application of formula (1) from 'Improved methods for the imputation
#     of missing data by nearest neighboure methods'(Gerhard Tutz, Shahla
#     Ramzan)
#     """
#     q = 2
#     xi_nan, xj_nan = np.isnan(xi), np.isnan(xj)
#     xi_mod, xj_mod = xi.copy(), xj.copy()
#     xi_mod[xi_nan] = 0
#     xj_mod[xj_nan] = 0
#     m_ij = ((~xi_nan)&(~xj_nan))
#     distance=(1/m_ij.sum(axis=0)*np.sum((np.abs(xi_mod-xj_mod)**q)*m_ij, axis=0))**(1/q)
#     return distance
#
# def min_max_scaler(A):
#     # minmax function that will limit the columns of A within the range [0,1]
#     mini=np.nanmin(A, axis=0)
#     maxi=np.nanmax(A, axis=0)
#     A_out=(A-mini)/(maxi-mini)
#     return A_out


def action_statistics(data, data_outliersremoved, data_raw, data_sampleholded, reporting_cycle=100):
    """ELIMINATE INSUFFICENT DATAPOINTS"""
    unique_icus = data_raw.icustay_id.unique()
    idx_include = []

    mechvent_1_mask = data_raw.mechvent==1
    
    print('Data raw idx_include:')
    for i, stay_id in enumerate(unique_icus):
        icu_mask = data_raw.icustay_id == stay_id
        mechvent_start = data_raw[mechvent_1_mask&icu_mask].index[0]
        starts = data_raw.iloc[mechvent_start].start_time
        mechvent_72hours = starts+pd.DateOffset(hours=72)
        current_idx = data_raw[icu_mask & (data_raw.start_time>=starts) & (data_raw.start_time<=mechvent_72hours)].index.values
        check_current_idx = np.ones(current_idx.shape)
        check_current_idx[1:] = np.diff(current_idx)
        current_idx = current_idx[check_current_idx.cumsum()==np.arange(1, current_idx.shape[0]+1, 1)]
        idx_include = np.concatenate([idx_include, current_idx])
        if i%reporting_cycle==0 and i>0:
            print(f'  Patient {i} finished.')

    ## Statistics after Outlier Removal
    # outliers removed

    print('Data outliers removed mechvent_0:')
    data_outliersremoved_tmp = data_outliersremoved.loc[idx_include, :]
    mechvent0_afteroutlierremoval = []
    for i, stay_id in enumerate(unique_icus):
        icu_mask = data_outliersremoved_tmp.icustay_id == stay_id
        if (data_outliersremoved_tmp[icu_mask].mechvent==0).any():
            mechvent0_afteroutlierremoval.append(stay_id)
        if i%reporting_cycle==0 and i>0:
            print(f'  Patient {i} finished.')

    print('Data sampleholded mechvent_0:')
    data_sampleholded_tmp = data_sampleholded.loc[idx_include, :]
    mechvent0_sampleholded = []
    for i, stay_id in enumerate(unique_icus):
        icu_mask = data_sampleholded_tmp.icustay_id == stay_id
        if (data_sampleholded_tmp[icu_mask].mechvent==0).any():
            mechvent0_sampleholded.append(stay_id)
        if i%reporting_cycle==0 and i>0:
            print(f'  Patient {i} finished.')

    # patients starting with a whole sequence of 1's and then ending with whole
    # sequence of 0's
    print('Data sampleholded erroneous_0:')
    mechvent0_sampleholded_erroneuos0 = []
    for i, stay_id in enumerate(unique_icus):
        icu_mask = data_sampleholded_tmp.icustay_id == stay_id
        data_crop = data_sampleholded_tmp[icu_mask].mechvent.fillna(0)
        if (data_crop==0).any(): # if there is an instance of 0
            ones_index = (data_crop[data_crop==1]).index.values
            zeros_index = (data_crop[data_crop==0]).index.values
            if all(np.concatenate([ones_index, zeros_index])==data_crop.index.values):
                mechvent0_sampleholded_erroneuos0.append(stay_id)
        if i%reporting_cycle==0 and i>0:
            print(f'  Patient {i} finished.')

    # patient with only one zero value between ones and during that time have a
    # measurement of Fio2, peep AND TVnormalized
    print('Data sampleholded alone zeros:')
    mechvent0_sampleholded_starting1ending0 = []
    for i, stay_id in enumerate(unique_icus):
        alone_zero_observed = False
        icu_mask = data_sampleholded_tmp.icustay_id == stay_id
        data_crop = data_sampleholded_tmp[icu_mask].mechvent.fillna(0)
        data_crop_features = data_sampleholded_tmp[icu_mask][['fio2', 'peep', 'TVnormalized']]
        alone_zeros = data_crop[data_crop==0].index.values
        if alone_zeros.size != 0:
            if alone_zeros[-1] == data_crop.index.values[-1]:
                alone_zeros = alone_zeros[:-1]
        if alone_zeros.size != 0:
            if alone_zeros[0] == data_crop.index.values[0]:
                alone_zeros = alone_zeros[1:]
        if alone_zeros.size != 0:
            for j in range(alone_zeros.size):
                between_ones = (data_crop.loc[alone_zeros[j]-1]==1)&(data_crop.loc[alone_zeros[j]+1]==1)
                data_avail = all(~data_crop_features.loc[alone_zeros[j]].isna())
                if between_ones & data_avail:
                    data_crop.loc[alone_zeros[j]] = 1
                    alone_zero_observed = True
        if alone_zero_observed:
            if (data_crop==0).any(): # if there is an instance of 0
                ones_index = (data_crop[data_crop==1]).index.values
                zeros_index = (data_crop[data_crop==0]).index.values
                if all(np.concatenate([ones_index, zeros_index])==data_crop.index.values):
                    mechvent0_sampleholded_starting1ending0.append(stay_id)
            elif (data_crop==1).all(): # if there are only 1s
                mechvent0_sampleholded_starting1ending0.append(stay_id) 
        if i%reporting_cycle==0 and i>0:
            print(f'  Patient {i} finished.')   

    # workaround not to ignore nans in mean calculations
    sampleholded_means = data_sampleholded_tmp[['icustay_id', 'fio2', 'peep', 'tidal_volume']].fillna(np.inf)
    sampleholded_means = sampleholded_means.groupby('icustay_id').mean().replace(np.inf, np.nan)
    all3features = sampleholded_means[(~sampleholded_means.isna()).product(axis=1)==1].index.values

    valid_icus = np.unique(np.concatenate([all3features, mechvent0_sampleholded_erroneuos0, mechvent0_sampleholded_starting1ending0]))

    different_TV = data_raw.TVnormalized!=data_outliersremoved.TVnormalized
    high_TV = data_raw.TVnormalized>25
    low_TV = data_raw.TVnormalized<0.1
    different_TV_not_na = ~(data_raw[different_TV].TVnormalized.isna())
    different_TV = different_TV[different_TV_not_na|high_TV|low_TV]
    different_TV = different_TV[different_TV==True].index.values
    data.loc[different_TV, 'TVnormalized'] = data_raw.loc[different_TV, 'TVnormalized']

    data_tmp = data.loc[idx_include, :]
    data_filtered = data_tmp[data_tmp.icustay_id.isin(valid_icus)]
    
    print('Data nonstandard measurement time diff:')
    time_diff = pd.concat([data_filtered.icustay_id, (data_filtered.start_time.diff()/pd.Timedelta(hours=1)).fillna(4).rename('hours_diff')], axis=1)
    uni_icu = time_diff.icustay_id.unique()
    time_diff.loc[:, 'hours_diff'] = np.where((time_diff.icustay_id.diff()>0), 4, time_diff.hours_diff)
    idx = []
    for i, stay_id in enumerate(uni_icu):
        idx_tmp = np.argwhere((time_diff.icustay_id==stay_id).values).flatten()
        if any(time_diff.iloc[idx_tmp, 1]!=4):
            not_4h = np.argwhere((time_diff.iloc[idx_tmp, 1]!=4).values).flatten()[0]
            nonstandard_diff = idx_tmp[not_4h:]
            idx = idx+list(nonstandard_diff)
        if i%reporting_cycle==0 and i>0:
            print(f'  Patient {i} finished.')
    data_filtered = data_filtered.drop(data_filtered.index[idx])
    return data_filtered
    
    
def correct_retrieved_table(data_filtered):
    """CONVERT TABLE into format required by later steps"""
    table_ori = data_filtered.copy(deep=True)
    hospmortandinouttimes = pd.read_csv('./data_in/hospmortandinouttimes.csv')

    unique_icustayid = table_ori.icustay_id.unique()

    hospmortandinouttimes = hospmortandinouttimes[hospmortandinouttimes.icustay_id.isin(unique_icustayid)].sort_values('icustay_id')

    unique_info = table_ori.groupby(['icustay_id', 'dischtime']).size().reset_index().loc[:, ['icustay_id', 'dischtime']]

    death_array = []
    for i, row in unique_info.iterrows():
        icu_id = row.icustay_id
        death = table_ori[table_ori.icustay_id==icu_id].deathtime.values[0]
        death_array.append(death)

    unique_info['deathtime'] = death_array
    block_tmp=[]
    last_measurement = []
    hosp_mort = []
    death_48 = []
    delay = []
    charttime = []
    for i in range(unique_info.shape[0]):
        total_records = np.sum(table_ori.icustay_id==unique_icustayid[i])
        block_tmp = block_tmp+list(np.arange(0, total_records)+1)
        last_measurement_tmp = table_ori[table_ori.icustay_id==unique_icustayid[i]].start_time
        last_measurement.append(last_measurement_tmp.iloc[-1])
        hosp_mort = hosp_mort+list(np.tile(hospmortandinouttimes.hospmort.iloc[i], total_records))                   
        if not pd.isna(pd.to_datetime(unique_info.iloc[i, 2])):
            under_48 = int(((unique_info.iloc[i,2]-pd.to_datetime(hospmortandinouttimes.outtime.iloc[i]))/pd.Timedelta(hours=1))<48)
            delay_val = (unique_info.iloc[i,2]-last_measurement[i])/pd.Timedelta(hours=1)
        else:
            under_48 = 0
            delay_val = (unique_info.iloc[i,1]-last_measurement[i])/pd.Timedelta(hours=1)
        death_48 = death_48+list(np.tile(under_48, total_records))
        delay = delay+list(np.tile(delay_val, total_records))   
        # In dataset description it is mentioned that posixtimes has been used
        # for ICU admission which is achieved by posixtime function here.
        in_time = pd.to_datetime(hospmortandinouttimes.iloc[i, 2])
        # to posix & utc timezone, for matlab compatibility
        in_time = in_time.replace(tzinfo=dt.timezone.utc).timestamp()
        charttime = charttime+list(np.tile(in_time, total_records))

    unique_info['last_measurement'] = last_measurement

    # final table definitions
    MIMIC_table = pd.DataFrame()
    MIMIC_table['bloc'] = block_tmp
    MIMIC_table['icustayid'] = table_ori.icustay_id.values
    MIMIC_table['charttime'] = charttime
    MIMIC_table['gender'] = np.where(table_ori.gender=='F', 1, 0)
    MIMIC_table['age'] = table_ori.first_admit_age.values
    MIMIC_table['elixhauser'] = table_ori.elixhauser_score.values
    MIMIC_table['re_admission'] = table_ori.icu_readm.values
    MIMIC_table['died_in'] = hosp_mort
    MIMIC_table['died_within_48h_of_out_time'] = death_48
    MIMIC_table['mortality_90d'] = table_ori.hospmort90day.values
    MIMIC_table['delay_end_of_record_and_discharge_or_death'] = delay
    MIMIC_table['Weight_kg'] = table_ori.weight.values
    MIMIC_table['GCS'] = table_ori.gcs.values
    MIMIC_table['HR'] = table_ori.heartrate.values
    MIMIC_table['SysBP'] = table_ori.sysbp.values
    MIMIC_table['MeanBP'] = table_ori.meanbp.values
    MIMIC_table['DiaBP'] = table_ori.diasbp.values
    MIMIC_table['RR'] = table_ori.resprate_set.values
    MIMIC_table['SpO2'] = table_ori.spo2.values
    MIMIC_table['Temp_C'] = table_ori.tempc.values
    MIMIC_table['FiO2_1'] = table_ori.fio2.values
    MIMIC_table['Potassium']=table_ori.potassium.values
    MIMIC_table['Sodium']=table_ori.sodium.values
    MIMIC_table['Chloride']=table_ori.chloride.values
    MIMIC_table['Glucose']=table_ori.glucose.values
    MIMIC_table['BUN']=table_ori.bun.values
    MIMIC_table['Creatinine']=table_ori.creatinine.values
    MIMIC_table['Magnesium']=table_ori.magnesium.values
    MIMIC_table['Calcium']=table_ori.calcium.values
    MIMIC_table['Ionised_Ca']=table_ori.ionizedcalcium.values
    MIMIC_table['CO2_mEqL']=table_ori.carbondioxide.values
    MIMIC_table['Total_bili']=table_ori.bilirubin.values
    MIMIC_table['Albumin']=table_ori.albumin.values
    MIMIC_table['Hb']=table_ori.hemoglobin.values
    MIMIC_table['WBC_count']=table_ori.wbc.values
    MIMIC_table['Platelets_count']=table_ori.platelet.values
    MIMIC_table['PTT']=table_ori.ptt.values
    MIMIC_table['PT']=table_ori.pt.values
    MIMIC_table['INR']=table_ori.inr.values
    MIMIC_table['Arterial_pH']=table_ori.ph.values
    MIMIC_table['paO2']=table_ori.pao2.values
    MIMIC_table['paCO2']=table_ori.paco2.values
    MIMIC_table['Arterial_BE']=table_ori.base_excess.values
    MIMIC_table['Arterial_lactate']=table_ori.lactate.values
    MIMIC_table['HCO3']=table_ori.bicarbonate.values
    MIMIC_table['mechvent']=table_ori.mechvent.values
    MIMIC_table['Shock_Index']=table_ori.shockindex.values
    MIMIC_table['PaO2_FiO2']=table_ori.pao2fio2ratio.values
    MIMIC_table['median_dose_vaso']=(0*(table_ori.vaso_total==0).astype(int)
                                    +30*((table_ori.vaso_total>0) & (table_ori.vaso_total<=50)).astype(int)
                                    +85*((table_ori.vaso_total>50) & (table_ori.vaso_total<=180)).astype(int)
                                    +320*((table_ori.vaso_total>180) & (table_ori.vaso_total<=530)).astype(int)
                                    +946*(table_ori.vaso_total>530).astype(int)).values
    MIMIC_table['max_dose_vaso']=table_ori.vaso_total.values
    MIMIC_table['input_total_tev']=table_ori.groupby('icustay_id')['iv_total'].transform(pd.Series.cumsum).values
    MIMIC_table['input_4hourly_tev']=table_ori.iv_total.values
    MIMIC_table['output_total']=table_ori.groupby('icustay_id')['urineoutput'].transform(pd.Series.cumsum).values
    MIMIC_table['output_4hourly']=table_ori.urineoutput.values
    MIMIC_table['cumulated_balance_tev']=MIMIC_table.output_total-MIMIC_table.input_total_tev
    MIMIC_table['SOFA']=table_ori.sofa.values
    MIMIC_table['SIRS']=table_ori.sirs.values
    MIMIC_table['PEEP']=table_ori.peep.values
    MIMIC_table['tidal_volume']=table_ori.tidal_volume.values
    MIMIC_table['tidal_volume_set']=table_ori.tidal_volume_set.values
    MIMIC_table['plat_pres']=table_ori.plateau_pressure.values
    MIMIC_table['idealbodyweight']=table_ori.ideal_body_weight_kg.values
    MIMIC_table['TVnormalized']=table_ori.TVnormalized.values

    return MIMIC_table