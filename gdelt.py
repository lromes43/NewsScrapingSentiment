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

today = datetime.date.today()
ten_years_ago = today -datetime.timedelta(days=10 * 365)

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

check = files_df.iloc[2]['text']

files_df['date'] = files_df['text'].str.split(".")
files_df['date']= files_df[0]


threshold = 20160226
for f in files_df: 





#need to be greater than 20160226.gkgcounts.csv.zip

threshold = 20160226

filtered_files = []

for f in files: 
    a_tag = f.find('a')
    if a_tag: 
        filename = a_tag.text.strip()
        try: 
            file_date = int(filename.split(".")[0])
            if file_date >= threshold: 
                filtered_files.append(filename)
        except ValueError: 
            pass

print(filtered_files)



pw = sync_playwright().start()

chrome = pw.chromium.launch(headless=False, args=[
        '--disable-http2', 
        '--no-sandbox'
    ])

page = chrome.new_page()
page.goto('http://data.gdeltproject.org/gkg/index.html')



filtered_files





#playwright

pw = sync_playwright().start()

chrome = pw.chromium.launch(headless=False, args=[
        '--disable-http2', 
        '--no-sandbox'
    ])

page = chrome.new_page()
page.goto('http://data.gdeltproject.org/gkg/index.html')


first_link = page.get_by_role('link').nth(2)
href = first_link.get_attribute('href')
print(href)



with page.expect_download() as download_info:
    first_link.click()

download = download_info.value

download.save_as("/Users/lukeromes/Desktop/NewsScrapingSentiment/Zips/20260222.gkg.csv.zip")
print(f"Downloaded to {download.path()}")



import pandas as pd

one = pd.read_csv(
    "/Users/lukeromes/Desktop/NewsScrapingSentiment/20260222.gkg.csv.zip",
    compression='zip', 
    sep='\t',          
    header=None          
)

print(one.head())