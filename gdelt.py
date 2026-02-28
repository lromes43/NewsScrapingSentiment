#gdelt scraping

import requests
from bs4 import BeautifulSoup
import pandas as pd
from playwright.sync_api import sync_playwright, Playwright
import re
import regex
import random
import time
import datetime
import numpy as np
import polars as pl
from zipfile import ZipFile
import glob
import os
import pyarrow.feather as feather
import gc
from joblib import Parallel, delayed


today = datetime.date.today()
ten_years_ago = today - datetime.timedelta(days=10 * 365)

#need to find the list number where ten_years ago date is 2/26/2016

url = 'http://data.gdeltproject.org/gkg/index.html'
raw_text = requests.get(url).text
souped = BeautifulSoup(raw_text, "html.parser")
souped
files = souped.find_all('li.ahref')

files = souped.select('li a')
data = []
for link in files:
    data.append({
        'text': link.get_text(strip=True),
        'url': link.get('href')
    })

files_df = pd.DataFrame(data).reset_index()
files_df['BinaryFilter'] = np.nan
files_df['date'] = np.nan
files_df = files_df[2:]


files_df['date'] = files_df['text'].str.extract(r'(\d{1,})')

threshold = 20160226.0
threshold2 = 20230323.0
threshold3 = 20221110.0
threshold4 = 20171031.0
files_df['date'] = pd.to_numeric(files_df['date'])


files_df['BinaryFilter'] = np.where(files_df['date'] >= threshold, 1, 0)

files_to_get = files_df[files_df['BinaryFilter'] == 1].copy()

files_to_get['newfilter'] = np.where(files_to_get['date'] >= threshold2, 0, 1)
files_to_get['newfilter2'] = np.where(files_to_get['date'] >= threshold3, 0, 1)
#files_to_get['newfilter3'] = np.where(files_to_get['date'] >= threshold3, 0, 1)

files_to_get = files_to_get[files_to_get['newfilter2'] ==1]

files_to_get.describe()

#need to be greater than 20160226.gkgcounts.csv.zip


#playwright


pw = sync_playwright().start()

chrome = pw.chromium.launch(headless=False, args=[
        '--disable-http2', 
        '--no-sandbox'
    ])

page = chrome.new_page()
page.goto('http://data.gdeltproject.org/gkg/index.html')




from zipfile import ZipFile
import os

os.makedirs("zips", exist_ok=True)
os.makedirs("feather", exist_ok=True)

#going to need to do parallel processing on this 



#save init feather files deleted once finished to save space on disk bc have the smaller versions rn

for link in files_to_get['text']: 
    print(f"Downloading: {link}")

    with page.expect_download() as download_info:
        page.get_by_text(link, exact=False).first.click()

    download = download_info.value

    zip_path = download.path()

    with ZipFile(zip_path) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pl.read_csv(
                f,
                separator="\t",
                has_header=False
            )

    df = df.select([
        df.columns[0],
        df.columns[4],
        df.columns[10]
    ]).rename({
        df.columns[0]: "DATE",
        df.columns[4]: "LOCATIONS",
        df.columns[10]: "SOURCEULRS"

    })

    feather_path = f"feather/{link.replace('.zip', '.parquet')}"
    df.write_ipc(feather_path)

    print(f"Saved {feather_path}")




#now that data is pulled will need to read them all in and concat them on top of one another in descending order of the dates and save to one_final df and save as a parquet to enable long 
#term storage


#little check to make sure correct data
data = pd.read_feather("/Users/lukeromes/Desktop/NewsScrapingSentiment/feather/20160302.gkg.csv.parquet")
data
data2 = pd.read_feather('/Users/lukeromes/Desktop/NewsScrapingSentiment/feather/20170105.gkg.csv.parquet')
data2['LOCATIONS']

#need to drop files that contain the ending.gkgcounts.csv.parquet
#rm *.gkgcounts.csv.parquet ran this in terminal but want to keep here to show


#go through and open every file and check in the Locations column if it has united states present
first_file = pd.read_feather("/Users/lukeromes/Desktop/NewsScrapingSentiment/feather/20160226.gkg.csv.parquet")
first_file = first_file.iloc[1:]

#psuedo code: 
    #iterate through everyrow, check if United States or one of the 50 States is present in the Locations col
    #if not present remove that row 


#checking for the first one 
us = ['United States', 'US', 'USA', 'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 
      'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 
      'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Montana', 
      'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota',
      'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas',
      'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin']


for i in range(1, len(first_file)):
    main_string = first_file['LOCATIONS'].iloc[i]
    substring_list = us

    if any(sub in main_string for sub in substring_list):
        print("At least one substring found at {i}")
    else:
        print("No substring was found at {i}.")
        first_file = first_file.drop(index=i)


first_file




#iterating through each file in the folder
#also trying to incorporate more psuedo code in my code so i know what i do when i go back 
#psuedo: 
    #read in each one, follow what we did above with the for loop
    #output new feather or parquet file with subset
    #clear ram/memory
    #go to next


import joblib
from joblib import parallel_config, Parallel, delayed
joblib.cpu_count()
#14 threads avalible 

os.chdir('/Users/lukeromes/Desktop/NewsScrapingSentiment/feather')
print("New working directory:", os.getcwd())

path = "/Users/lukeromes/Desktop/NewsScrapingSentiment/feather"



def producer():
    for p in glob.glob(os.path.join(path, '*.parquet')):
        pathname1 = os.path.basename(p)
        pathname = re.sub('\.parquet','', pathname1)
        df = pd.read_feather(pathname1)
        df = df.iloc[1:]
        for i in range(1, len(p)):
            main_string = df['LOCATIONS'].iloc[i]
            substring_list = us

            if any(sub in main_string for sub in substring_list):
                print(f"Match  good at column {i}")
            else:
                print(f"No match so removed column {i}.")
                df = df.drop(index=i)
            
            df.to_parquet(f"/Users/lukeromes/Desktop/NewsScrapingSentiment/cleaneddata/{pathname}.parquet")


out = Parallel(n_jobs=12, verbose=100, pre_dispatch='1.5*n_jobs')(
    delayed()(i) for i in producer())

#check to see how it worked
#decreased original size of 140.75 GB (3633 files) to 57.73 (3631 files)


#psuedocode

#so basically just need to iterate through every row and click the source url, do a get request, extract the html and from there can take out the paragraohs and title and save the text for each to a .txt file


#start with just pulling first row of first file

check_data = pd.read_parquet("/Users/lukeromes/Desktop/NewsScrapingSentiment/cleaneddata/20160226.gkg.csv.parquet")
check_data


first_row_date = check_data.iloc[0]['DATE']
first_link= check_data.iloc[0]['SOURCEULRS']
request = requests.get(first_link)
soup = BeautifulSoup(request.text, 'html.parser')


p_tags = soup.find_all('p')
p_tags

all_text = [p.text.strip() for p in soup.find_all('p')]
combined_text = " ".join([text.strip() for text in all_text])
df = pd.DataFrame([{'Date':'N/A', 
                   'Text':'N/A'}])
df['Date'] = first_row_date
df['Text'] = combined_text

first_link
new = re.sub('http://', '', first_link)
new

clean_id = re.sub(r'\.[^.]+$', '', new)
clean_id = re.sub('/', '_', clean_id)
clean_id = re.sub('\.', '_', clean_id)

df.to_csv(f'/Users/lukeromes/Desktop/NewsScrapingSentiment/textfiles/{clean_id}text.csv', index=False, encoding='utf-8')




#now going to try with just the first file and iterate through the whole thing
#need to make sure to do joblib for parallel processing



#concatenating each together


def appending(p):
    d = "/Users/lukeromes/Desktop/NewsScrapingSentiment/cleaneddata"
    files = []
    df = pd.read_parquet(p)

    files.append({
            'Date': df['DATE'],
            'URL': df['SOURCEULRS'],
            'Locations': df['LOCATIONS']
        })

    return df
    
results = Parallel(n_jobs=12)(delayed(appending)(p) for p in glob.glob(os.path.join('"/Users/lukeromes/Desktop/NewsScrapingSentiment/cleaneddata"', '*.parquet')))
print(results)






# Convert the whole thing to a single DataFrame at the very end
# final_df = pd.DataFrame(files)


df['domain'] = df['URL'].str.extract(r'https?://(?:www\.)?([^/]+)')
df['domain'].unique()

#once have all of this will go through and pull the data from each link doing like get requests then will soup it to find all of the paragraphs and what not

#run through some topic models to do sentiment then apply these to stock project see which ones help the most if any as well as can we get entire sentiment from just
#article title or do we need to read the whole thing? Does this change throughout the years as well. 

#and extract the key themes but make sure to keep the date, want to incorporate this into the stacked ML model I have


from pyarrow import csv

read_options = csv.ReadOptions(
    block_size=1e+7,  
    use_threads =True
)

table = csv.read_csv("/Users/lukeromes/Desktop/NewsScrapingSentiment/final_combined_data.csv", read_options=read_options)

df = table.to_pandas()






