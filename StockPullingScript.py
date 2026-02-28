#took from my stock ML project from fall

import sys
sys.path.append('/Users/lukeromes/Desktop/NewsScrapingSentiment/Functions')
import sp500pipeline
import pandas as pd
from datetime import date




date1_in_range = "2016-02-26" 
date2_in_range = "2016-03-03"
sp500pipeline.run_sp500_pipeline(start_date = date1_in_range , end_date = date2_in_range)



data_path = "/Users/lukeromes/Desktop/Sp500Project/Data/FINALSP500Data.feather"
train_end_date = pd.to_datetime("2025-11-12").tz_localize(None)
test_start_date = pd.to_datetime("2025-11-13").tz_localize(None)

