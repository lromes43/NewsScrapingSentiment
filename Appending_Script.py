from joblib import Parallel, delayed
import glob
import os
import pandas as pd
import pyarrow


def appending(p):
    cols = ['DATE', 'LOCATIONS', 'SOURCEULRS']
    
    df = pd.read_parquet(p, columns=cols, engine='pyarrow')

    df['DATE'] = pd.to_datetime(df['DATE'])
    df['LOCATIONS'] = df['LOCATIONS'].astype('category')
    df['SOURCEULRS'] = df['SOURCEULRS'].astype('string[pyarrow]')

    return df

results = Parallel(n_jobs=-1, verbose=10, backend="loky")(
    delayed(appending)(p) for p in glob.glob(os.path.join('/Users/lukeromes/Desktop/NewsScrapingSentiment/cleaneddata', '*.parquet'))
)

final_df = pd.concat(results, ignore_index=True)

final_df.to_parque("final_combined_data.parquet")
