import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import os
from joblib import Parallel, delayed
import shutil
import polars as pl



def process_single_article(row_data):
    
    free_gb = shutil.disk_usage('/Users/lukeromes/').free // (2**30)
    
    if free_gb < 90:
        print(f"Out of stoage")
        return "Low Disk Space Error" 

    first_row_date = row_data['DATE']
    first_link = row_data['SOURCEULRS']
    
    try: 
        request = requests.get(first_link, timeout=10) 
        if request.status_code != 200: 
            return f"Failed: {request.status_code}" 
            
        soup = BeautifulSoup(request.text, 'html.parser')
        all_text = [p.text.strip() for p in soup.find_all('p')]
        combined_text = " ".join([text.strip() for text in all_text])
        
        new = re.sub('https?://', '', first_link) 
        clean_id = re.sub(r'\.[^.]+$', '', new)
        clean_id = re.sub(r'[./]', '_', clean_id) 

        save_path = f'/Users/lukeromes/Desktop/NewsScrapingSentiment/textfiles/{clean_id}text.csv'
        pd.DataFrame([{'Date': first_row_date, 'Text': combined_text}]).to_csv(
            save_path, index=False, encoding='utf-8'
        )
        return "Success"

    except Exception as e:
        return f"Error: {str(e)}"
    


if __name__ == "__main__":
    target_dir = '/Users/lukeromes/Desktop/NewsScrapingSentiment/textfiles/'
    os.makedirs(target_dir, exist_ok=True)
    
    lazy_df = pl.scan_parquet('/Users/lukeromes/Desktop/NewsScrapingSentiment/final_combined_data.parquet')
    

    batch_size = 5000 
    

    total_rows = lazy_df.select(pl.len()).collect().item()
    
    for offset in range(0, total_rows, batch_size):

        batch = lazy_df.slice(offset, batch_size).collect()
        rows = batch.to_dicts() # Much safer on a small slice
        
        print(f"Processing batch: {offset} to {offset + batch_size}")
        
        results = Parallel(n_jobs=12, verbose=5, backend="loky")(
            delayed(process_single_article)(row) for row in rows
        )


