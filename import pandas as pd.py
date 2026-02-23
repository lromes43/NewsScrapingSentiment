import pandas as pd
from bs4 import BeautifulSoup
import requests
from PIL import Image
from io import BytesIO
import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False) 
        page = await browser.new_page()

        with open("noaa_requests.txt", "a") as f:
        
            page.on("request", lambda request: f.write(f"{request.url}\n"))

            print("Loading NOAA... Logging all requests to noaa_requests.txt")
            await page.goto("https://www.nesdis.noaa.gov/imagery/interactive-maps/earth-real-time")
            

            await page.wait_for_selector("canvas", timeout=60000)

            while True:
                await page.mouse.move(500, 500)
                await page.mouse.down()
                await page.mouse.move(800, 500, steps=20)
                await page.mouse.up()
                
                await asyncio.sleep(2)
                print("Captured a rotation's worth of requests...")

try:
    asyncio.run(run())
except KeyboardInterrupt: 
    print('stopped')



requestss = pd.read_csv("/Users/lukeromes/Desktop/NewsScrapingSentiment/noaa_requests.txt")
requestss.iloc[0]
response = requests.get(requestss.iloc[0], headers={':authority': 'satellitemaps.nesdis.noaa.gov', 
                                                    'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8'})
response


link = "https://satellitemaps.nesdis.noaa.gov/arcgis/rest/services/MERGEDGC_current/ImageServer/tile/2/3/3"
response = requests.get(link)
if response.status_code == 200:
    # Open the image from the in-memory binary data using Pillow (PIL)
    img = Image.open(BytesIO(response.content))

    # Display the image (this will open an image viewer on your system)
    img.show()

    print("Image opened successfully.")
else:
    print(f"Failed to retrieve image. Status code: {response.status_code}")
