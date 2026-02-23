#Yahoo News Scrapping Script
import pandas as pd
from playwright.sync_api import sync_playwright, Playwright
import re
import regex
import random
import time
pw = sync_playwright().start()

chrome = pw.chromium.launch(headless=False, args=[
        '--disable-http2', 
        '--no-sandbox'
    ])

page = chrome.new_page()

page.goto('https://web.archive.org/web/20260220225532/https://finance.yahoo.com/markets/stocks/most-active/', 
          wait_until='domcontentloaded', 
          timeout=90000)
page.wait_for_load_state('load')
company = page.locator('.yf-grb3qw .tableContainer .table-container .yf-1og7bvd').locator('.row').nth(0).locator('.ticker').inner_text()
page.locator('.yf-grb3qw .tableContainer .table-container .yf-1og7bvd').locator('.row').nth(0).locator('.ticker').click()
company_page = page.url
page.wait_for_load_state('networkidle', timeout=90000)


number_articles = page.locator('.yf-11mm4ul').locator('.content').locator('h3').count()

#article 1

init_link =page.locator('.yf-11mm4ul').locator('.content').nth(0).locator('.subtle-link').get_attribute('href')
final_link = regex.findall(r'(?<=https://web.archive.org/web/\d+/)(.*)', init_link)[0]
article_title = page.locator('.yf-11mm4ul').locator('.content').nth(0).inner_text()

try:
    page.goto(final_link, wait_until="domcontentloaded", timeout=90000)
    page.wait_for_selector("article, .caas-body", timeout=30000)
except:
    print("Article snapshot unavailable, skipping")

page.go_back()

current_row = {'Company': 'NVDA',
               'Article_Title': article_title, 
               'Text': 'N/A'}

current_row['text'] = page.locator('.bodyItems-wrapper').locator('p').all_inner_texts()
page.goto(company_page)






#trying to loop through articles now 

rest = random.uniform(.1,.3)

page.goto('https://web.archive.org/web/20260220225532id_/https://finance.yahoo.com/markets/stocks/most-active/', 
          wait_until='domcontentloaded', 
          timeout=130000)
page.wait_for_load_state('load')
company = page.locator('.yf-grb3qw .tableContainer .table-container .yf-1og7bvd').locator('.row').nth(0).locator('.ticker').inner_text()
page.locator('.yf-grb3qw .tableContainer .table-container .yf-1og7bvd').locator('.row').nth(0).locator('.ticker').click()
company_page = page.url
page.wait_for_load_state('networkidle', timeout=130000)
complete_text = []
number_articles = page.locator('.yf-11mm4ul').locator('.content').locator('h3').count()
for a in range(0, number_articles):
    article_container = page.locator('.yf-11mm4ul').locator('.content').nth(a)
    init_link = article_container.locator('.subtle-link').get_attribute('href')
    article_title = article_container.inner_text()
    matches = regex.findall(r'(?<=https://web.archive.org/web/\d+/)(.*)', init_link)
    final_link = matches[0] if matches else init_link
    current_row = {'Company': 'NVDA', 'Article_Title': article_title, 'Text': 'N/A'}
    time.sleep(random.uniform(.1, .5))
    try:
        page.goto(final_link, wait_until="domcontentloaded", timeout=90000)
        page.wait_for_selector('.bodyItems-wrapper, .caas-body', timeout=15000)
        paragraphs = page.locator('.bodyItems-wrapper p, .caas-body p').all_inner_texts()
        current_row['Text'] = " ".join(paragraphs)
    except Exception as e:
        print(f"Skipping article {a}: {e}")
    complete_text.append(current_row)
    page.goto(company_page, wait_until="domcontentloaded", timeout=90000)
    page.wait_for_selector('.yf-11mm4ul', timeout=30000)
    time.sleep(random.uniform(90,120))

pd.DataFrame(complete_text)



import requests
link = 'https://guce.yahoo.com/consent?brandType=nonEu&gcrumb=blfTaFo&done=https%3A%2F%2Ffinance.yahoo.com%2F'
requests.get(link)





#fox business 

import pandas as pd
from playwright.sync_api import sync_playwright, Playwright
import re
import regex
pw = sync_playwright().start()

chrome = pw.chromium.launch(headless=False)

page = chrome.new_page()

page.goto('https://web.archive.org/web/20260216025519/https://www.foxbusiness.com/category/stocks')
