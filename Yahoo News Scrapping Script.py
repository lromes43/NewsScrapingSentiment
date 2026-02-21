#Yahoo News Scrapping Script
import pandas as pd
from playwright.sync_api import sync_playwright, Playwright
import re
import regex
pw = sync_playwright().start()

chrome = pw.chromium.launch(headless=False)

page = chrome.new_page()

page.goto('https://web.archive.org/web/20260220102145/https://finance.yahoo.com/markets/stocks/most-active/')
company = page.locator('.yf-grb3qw .tableContainer .table-container .yf-1og7bvd').locator('.row').nth(0).locator('.ticker').inner_text()
page.locator('.yf-grb3qw .tableContainer .table-container .yf-1og7bvd').locator('.row').nth(0).locator('.ticker').click()
page.wait_for_load_state('load')


number_articles = page.locator('.yf-11mm4ul').locator('.content').locator('h3').count()

#article 1

init_link =page.locator('.yf-11mm4ul').locator('.content').nth(0).locator('.subtle-link').get_attribute('href')
final_link = regex.findall(r'(?<=https://web.archive.org/web/\d+/)(.*)', init_link)[0]
article_title = page.locator('.yf-11mm4ul').locator('.content').nth(1).inner_text()

try:
    page.goto(final_link, wait_until="domcontentloaded", timeout=20000)
    page.wait_for_selector("article, .caas-body", timeout=10000)
except:
    print("Article snapshot unavailable, skipping")

page.go_back()

current_row = {'Company': 'NVDA',
               'Article_Title': article_title, 
               'Text': 'N/A'}

current_row['text'] = page.locator('.bodyItems-wrapper').locator('p').all_inner_texts()