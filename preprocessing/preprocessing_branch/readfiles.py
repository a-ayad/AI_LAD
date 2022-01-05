import pandas as pd

data_raw = pd.read_pickle(r'pickle/data_raw.pkl')
data_med = pd.read_pickle(r'pickle/data_med.pkl')
smphold = pd.read_pickle(r'pickle/data_sampleholded.pkl')
outremove = pd.read_pickle(r'pickle/data_outliersremoved.pkl')

print("Done")