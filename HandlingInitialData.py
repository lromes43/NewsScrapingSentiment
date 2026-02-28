import pandas as pd 
import polars as pl
import glob 
import os 
import tqdm
import numpy as np
import textwrap
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from flair.nn import Classifier
from flair.data import Sentence
from transformers import pipeline
from flair.data import Sentence
from flair.models import TextClassifier

#need to check example file before iterating through everyone: 
ex = pd.read_csv("/Users/lukeromes/Desktop/NewsScrapingSentiment/textfiles/2paragraphs_com_2016_02_american-express-makes-soap-at-nbc_text.csv")
ex
path = '/Users/lukeromes/Desktop/NewsScrapingSentiment/textfiles'


df = pl.scan_csv(os.path.join(path, "*.csv")).collect()
df.write_parquet("text.parquet")

#initial data exploration and removing the empties
df = pl.read_parquet("/Users/lukeromes/Desktop/NewsScrapingSentiment/text.parquet")
df = df.to_pandas()
df

df['Text'] = df['Text'].replace('', np.nan)
df['Text'] = df['Text'].replace('None', np.nan)
df.isna().sum() # gives 51017 null articles

df = df.dropna().reset_index()
df = df.drop('index', axis = 1)


df.iloc[279972]['Text']

df['exception_flag'] = df['Text'].str.contains('forbidden').astype(int)
df = df[df['exception_flag']==0]
df



#will extract basic document polarity first using vader 
#will need to check out length of each document and not make too long as computational cost can add up quickly 
#trimming to 10000 characters

analyzer = SentimentIntensityAnalyzer()
df['Trimmed_Text'] = df['Text'].str.slice(0, 10000)
df = df.drop_duplicates()
df

df['vader_polarity'] = df['Trimmed_Text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])




daily_averages = df.groupby('Date')['vader_polarity'].mean().reset_index()
daily_averages = daily_averages.rename(columns={'vader_polarity':'mean_polarity'})
daily_averages
df = df.drop('Daily_Polarity_Score', axis=1)
df

merged_df = pd.merge(daily_averages, df, how = 'inner', on = 'Date')
merged_df['vader_polarity_classification'] = np.nan

for i in range(len(merged_df)): 
    if merged_df['vader_polarity'].iloc[i] < 0.5: 
        merged_df['vader_polarity_classification'].iloc[i] = 'Very_Negative'
    elif (merged_df['vader_polarity'].iloc[i] < 0 and merged_df['vader_polarity'].iloc[i] > 0.5): 
        merged_df['vader_polarity_classification'].iloc[i] = 'Slight_Negative'
    elif merged_df['vader_polarity'].iloc[i] == 0: 
        merged_df['vader_polarity_classification'] = 'Neutral'
    elif (merged_df['vader_polarity'].iloc[i] > 0 and merged_df['vader_polarity'].iloc[i] < 0.5): 
        merged_df['vader_polarity_classification'].iloc[i] = 'Slight_Positive'
    elif merged_df['vader_polarity'].iloc[i] > .5: 
        merged_df['vader_polarity_classification'].iloc[i] = 'Very_Positive'



#going back and doing some additional cleaning I missed initially

merged_df['Trimmed_Text'] = merged_df['Trimmed_Text'].astype(str).str.strip()
merged_df.replace('', np.nan, inplace=True)

merged_df = merged_df.dropna(subset=['Trimmed_Text']).reset_index(drop=True)

merged_df = merged_df.drop_duplicates()
merged_df


#flair 


classifier = TextClassifier.load('en-sentiment')


sentence_list = [Sentence(text) for text in merged_df['Trimmed_Text']]

print("Starting predictions...")
classifier.predict(sentence_list, mini_batch_size=32, verbose=True)


merged_df['Flair_Label'] = [s.labels[0].value if s.labels else "UNKNOWN" for s in sentence_list]
merged_df['Flair_Score'] = [s.labels[0].score if s.labels else 0.0 for s in sentence_list]

 
merged_df = merged_df.drop(['NB_Classification', 'NB_Polarity'], axis=1)
merged_df



##Pre-trained Model regarding financial news sentiment I found on hugging face
#link: https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis


pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",truncation=True,max_length=512)

for i in range(len(merged_df)): 
    text = merged_df['Trimmed_Text'].iloc[i]
    result = pipe(text)
    merged_df['Fin_Sent_Label'] = result[0]['label']
    merged_df['Fin_Sent_Score'] = result[0]['score']
    if i % 1000 == 0: 
        print(i)
    else:
        continue


merged_df



#other hugging face model


pipe2= pipeline("text-classification", model="finiteautomata/beto-headlines-sentiment-analysis", truncation = True, max_length = 512)


for i in range(len(merged_df)): 
    text = merged_df['Trimmed_Text'].iloc[i]
    result = pipe2(text)
    merged_df['News_Headlines_Label'] = result[0]['label']
    merged_df['News_Headlines_Score'] = result[0]['score']
    if i % 1000 == 0: 
        print(i)
    else:
        continue

merged_df



merged_df.columns

merged_df[['vader_polarity', 'vader_polarity_classification', 'Flair_Label',
       'Flair_Score', 'Fin_Sent_Label', 'Fin_Sent_Score',
       'News_Headlines_Label', 'News_Headlines_Score']]

merged_df['vader_polarity_classification']

for i in range(len(merged_df)): 
    if merged_df['vader_polarity'].iloc[i] > 0.0: 
        merged_df['vader_polarity_classification'] = 'positive'
    elif merged_df['vader_polarity'].iloc[i] == 0.0:
        merged_df['vader_polarity_classification'].iloc[i] = 'neutral'
    elif merged_df['vader_polarity'].iloc[i] < 0.0: 
        merged_df['vader_polarity_classification'].iloc[i] = 'negative'

for i in range(len(merged_df)): 
    if merged_df['Flair_Label'].iloc[i] == 'NEGATIVE': 
        merged_df['Flair_Score'].iloc[i] = merged_df['Flair_Score'].iloc[i] * -1



merged_df


#eventually have a flag that checks if one of the stocks in sp500 is present in the Text col or the word like finance or stock market or wall street


stocks = pd.read_feather("/Users/lukeromes/Desktop/SP500Project/Data/FINALSP500Data.feather")
stocks.columns.to_list()


tickers = stocks['Ticker'].unique()
ticker_list = list(tickers)

#created a dictionary of every single sp500 company and filtered to see if the text had at least one company present, if so would get a flag

ticker_mapping = {
    'A': 'Agilent Technologies', 'AAPL': 'Apple Inc.', 'ABBV': 'AbbVie', 'ABNB': 'Airbnb',
    'ABT': 'Abbott Laboratories', 'ACGL': 'Arch Capital Group', 'ACN': 'Accenture',
    'ADBE': 'Adobe Inc.', 'ADI': 'Analog Devices', 'ADM': 'Archer-Daniels-Midland',
    'ADP': 'Automatic Data Processing', 'ADSK': 'Autodesk', 'AEE': 'Ameren',
    'AEP': 'American Electric Power', 'AES': 'AES Corporation', 'AFL': 'Aflac',
    'AIG': 'American International Group', 'AIZ': 'Assurant', 'AJG': 'Arthur J. Gallagher & Co.',
    'AKAM': 'Akamai Technologies', 'ALB': 'Albemarle Corporation', 'ALGN': 'Align Technology',
    'ALL': 'Allstate', 'ALLE': 'Allegion', 'AMAT': 'Applied Materials',
    'AMCR': 'Amcor', 'AMD': 'Advanced Micro Devices', 'AME': 'AMETEK Inc.',
    'AMGN': 'Amgen', 'AMP': 'Ameriprise Financial', 'AMT': 'American Tower',
    'AMTM': 'Amentum', 'AMZN': 'Amazon.com Inc.', 'ANET': 'Arista Networks',
    'AON': 'Aon plc', 'AOS': 'A. O. Smith', 'APA': 'APA Corporation',
    'APD': 'Air Products and Chemicals', 'APH': 'Amphenol', 'APTV': 'Aptiv',
    'ARE': 'Alexandria Real Estate Equities', 'ATO': 'Atmos Energy', 'AVB': 'AvalonBay Communities',
    'AVGO': 'Broadcom Inc.', 'AVY': 'Avery Dennison', 'AWK': 'American Water Works',
    'AXON': 'Axon Enterprise', 'AXP': 'American Express', 'AZO': 'AutoZone',
    'BA': 'Boeing', 'BAC': 'Bank of America', 'BALL': 'Ball Corporation',
    'BAX': 'Baxter International', 'BBY': 'Best Buy', 'BDX': 'Becton Dickinson',
    'BEN': 'Franklin Resources', 'BG': 'Bunge Global SA', 'BIIB': 'Biogen',
    'BK': 'The Bank of New York Mellon', 'BKNG': 'Booking Holdings', 'BKR': 'Baker Hughes',
    'BLDR': 'Builders FirstSource', 'BLK': 'BlackRock', 'BMY': 'Bristol-Myers Squibb',
    'BR': 'Broadridge Financial Solutions', 'BRO': 'Brown & Brown', 'BSX': 'Boston Scientific',
    'BWA': 'BorgWarner', 'BX': 'Blackstone', 'BXP': 'BXP Inc.',
    'C': 'Citigroup', 'CAG': 'Conagra Brands', 'CAH': 'Cardinal Health',
    'CARR': 'Carrier Global', 'CAT': 'Caterpillar Inc.', 'CB': 'Chubb Limited',
    'CBOE': 'Cboe Global Markets', 'CBRE': 'CBRE Group', 'CCI': 'Crown Castle',
    'CCL': 'Carnival Corporation', 'CDNS': 'Cadence Design Systems', 'CDW': 'CDW Corporation',
    'CE': 'Celanese', 'CEG': 'Constellation Energy', 'CF': 'CF Industries',
    'CFG': 'Citizens Financial Group', 'CHD': 'Church & Dwight', 'CHRW': 'C.H. Robinson',
    'CHTR': 'Charter Communications', 'CI': 'The Cigna Group', 'CINF': 'Cincinnati Financial',
    'CL': 'Colgate-Palmolive', 'CLX': 'The Clorox Company', 'CMCSA': 'Comcast',
    'CME': 'CME Group', 'CMG': 'Chipotle Mexican Grill', 'CMI': 'Cummins',
    'CMS': 'CMS Energy', 'CNC': 'Centene Corporation', 'CNP': 'CenterPoint Energy',
    'COF': 'Capital One', 'COO': 'The Cooper Companies', 'COP': 'ConocoPhillips',
    'COR': 'Cencora', 'COST': 'Costco', 'CPAY': 'Corpay',
    'CPB': 'Campbell Soup Company', 'CPRT': 'Copart', 'CPT': 'Camden Property Trust',
    'CRL': 'Charles River Laboratories', 'CRM': 'Salesforce', 'CRWD': 'CrowdStrike',
    'CSCO': 'Cisco', 'CSGP': 'CoStar Group', 'CSX': 'CSX Corporation',
    'CTAS': 'Cintas', 'CTRA': 'Coterra', 'CTSH': 'Cognizant',
    'CTVA': 'Corteva', 'CVS': 'CVS Health', 'CVX': 'Chevron Corporation',
    'CZR': 'Caesars Entertainment', 'D': 'Dominion Energy', 'DAL': 'Delta Air Lines',
    'DAY': 'Dayforce', 'DD': 'DuPont', 'DE': 'John Deere',
    'DECK': 'Deckers Brands', 'DELL': 'Dell Technologies', 'DG': 'Dollar General',
    'DGX': 'Quest Diagnostics', 'DHI': 'D.R. Horton', 'DHR': 'Danaher Corporation',
    'DIS': 'Walt Disney Company', 'DLR': 'Digital Realty', 'DLTR': 'Dollar Tree',
    'DOC': 'Healthpeak Properties', 'DOV': 'Dover Corporation', 'DOW': 'Dow Inc.',
    'DPZ': 'Domino\'s', 'DRI': 'Darden Restaurants', 'DTE': 'DTE Energy',
    'DUK': 'Duke Energy', 'DVA': 'DaVita Inc.', 'DVN': 'Devon Energy',
    'DXCM': 'Dexcom', 'EA': 'Electronic Arts', 'EBAY': 'eBay',
    'ECL': 'Ecolab', 'ED': 'Consolidated Edison', 'EFX': 'Equifax',
    'EG': 'Everest Group', 'EIX': 'Edison International', 'EL': 'Estée Lauder Companies',
    'ELV': 'Elevance Health', 'EMN': 'Eastman Chemical', 'EMR': 'Emerson Electric',
    'ENPH': 'Enphase Energy', 'EOG': 'EOG Resources', 'EPAM': 'EPAM Systems',
    'EQIX': 'Equinix', 'EQR': 'Equity Residential', 'EQT': 'EQT Corporation',
    'ERIE': 'Erie Indemnity', 'ES': 'Eversource', 'ESS': 'Essex Property Trust',
    'ETN': 'Eaton Corporation', 'ETR': 'Entergy', 'EVRG': 'Evergy',
    'EW': 'Edwards Lifesciences', 'EXC': 'Exelon', 'EXPD': 'Expeditors International',
    'EXPE': 'Expedia Group', 'EXR': 'Extra Space Storage', 'F': 'Ford Motor Company',
    'FANG': 'Diamondback Energy', 'FAST': 'Fastenal', 'FCX': 'Freeport-McMoRan',
    'FDS': 'FactSet', 'FDX': 'FedEx', 'FE': 'FirstEnergy',
    'FFIV': 'F5 Inc.', 'FICO': 'Fair Isaac', 'FIS': 'FIS',
    'FITB': 'Fifth Third Bank', 'FMC': 'FMC Corporation', 'FOX': 'Fox Corporation (Class B)',
    'FOXA': 'Fox Corporation (Class A)', 'FRT': 'Federal Realty', 'FSLR': 'First Solar',
    'FTNT': 'Fortinet', 'FTV': 'Fortive', 'GD': 'General Dynamics',
    'GDDY': 'GoDaddy', 'GE': 'GE Aerospace', 'GEHC': 'GE HealthCare',
    'GEN': 'Gen Digital', 'GEV': 'GE Vernova', 'GILD': 'Gilead Sciences',
    'GIS': 'General Mills', 'GL': 'Globe Life', 'GLW': 'Corning Inc.',
    'GM': 'General Motors', 'GNRC': 'Generac', 'GOOG': 'Alphabet Inc. (Class C)',
    'GOOGL': 'Alphabet Inc. (Class A)', 'GPC': 'Genuine Parts Company', 'GPN': 'Global Payments',
    'GRMN': 'Garmin', 'GS': 'Goldman Sachs', 'GWW': 'W.W. Grainger',
    'HAL': 'Halliburton', 'HAS': 'Hasbro', 'HBAN': 'Huntington Bancshares',
    'HCA': 'HCA Healthcare', 'HD': 'Home Depot', 'HIG': 'The Hartford',
    'HII': 'Huntington Ingalls Industries', 'HLT': 'Hilton Worldwide', 'HOLX': 'Hologic',
    'HON': 'Honeywell', 'HPE': 'Hewlett Packard Enterprise', 'HPQ': 'HP Inc.',
    'HRL': 'Hormel Foods', 'HSIC': 'Henry Schein', 'HST': 'Host Hotels & Resorts',
    'HSY': 'The Hershey Company', 'HUBB': 'Hubbell Incorporated', 'HUM': 'Humana',
    'HWM': 'Howmet Aerospace', 'IBM': 'IBM', 'ICE': 'Intercontinental Exchange',
    'IDXX': 'IDEXX Laboratories', 'IEX': 'IDEX Corporation', 'IFF': 'International Flavors & Fragrances',
    'INCY': 'Incyte', 'INTC': 'Intel', 'INTU': 'Intuit',
    'INVH': 'Invitation Homes', 'IP': 'International Paper', 'IQV': 'IQVIA',
    'IR': 'Ingersoll Rand', 'IRM': 'Iron Mountain', 'ISRG': 'Intuitive Surgical',
    'IT': 'Gartner', 'ITW': 'Illinois Tool Works', 'IVZ': 'Invesco',
    'J': 'Jacobs Solutions', 'JBHT': 'J.B. Hunt', 'JBL': 'Jabil',
    'JCI': 'Johnson Controls', 'JKHY': 'Jack Henry & Associates', 'JNJ': 'Johnson & Johnson',
    'JPM': 'JPMorgan Chase', 'K': 'Kellanova', 'KDP': 'Keurig Dr Pepper',
    'KEY': 'KeyCorp', 'KEYS': 'Keysight', 'KHC': 'The Kraft Heinz Company',
    'KIM': 'Kimco Realty', 'KKR': 'KKR & Co. Inc.', 'KLAC': 'KLA Corporation',
    'KMB': 'Kimberly-Clark', 'KMI': 'Kinder Morgan', 'KMX': 'CarMax',
    'KO': 'Coca-Cola Company', 'KR': 'Kroger', 'KVUE': 'Kenvue',
    'L': 'Loews Corporation', 'LDOS': 'Leidos', 'LEN': 'Lennar',
    'LH': 'Labcorp', 'LHX': 'L3Harris', 'LIN': 'Linde plc',
    'LKQ': 'LKQ Corporation', 'LLY': 'Eli Lilly and Company', 'LMT': 'Lockheed Martin',
    'LNT': 'Alliant Energy', 'LOW': 'Lowe\'s', 'LRCX': 'Lam Research',
    'LULU': 'Lululemon Athletica', 'LUV': 'Southwest Airlines', 'LVS': 'Las Vegas Sands',
    'LW': 'Lamb Weston', 'LYB': 'LyondellBasell', 'LYV': 'Live Nation Entertainment',
    'MA': 'Mastercard', 'MAA': 'Mid-America Apartment Communities', 'MAR': 'Marriott International',
    'MAS': 'Masco', 'MCD': 'McDonald\'s', 'MCHP': 'Microchip Technology',
    'MCK': 'McKesson', 'MCO': 'Moody\'s Corporation', 'MDLZ': 'Mondelēz International',
    'MDT': 'Medtronic', 'MET': 'MetLife', 'META': 'Meta Platforms',
    'MGM': 'MGM Resorts', 'MHK': 'Mohawk Industries', 'MKC': 'McCormick & Company',
    'MKTX': 'MarketAxess', 'MLM': 'Martin Marietta Materials', 'MMC': 'Marsh McLennan',
    'MMM': '3M', 'MNST': 'Monster Beverage', 'MO': 'Altria',
    'MOH': 'Molina Healthcare', 'MOS': 'The Mosaic Company', 'MPC': 'Marathon Petroleum',
    'MPWR': 'Monolithic Power Systems', 'MRK': 'Merck & Co.', 'MRNA': 'Moderna',
    'MS': 'Morgan Stanley', 'MSCI': 'MSCI Inc.', 'MSFT': 'Microsoft',
    'MSI': 'Motorola Solutions', 'MTB': 'M&T Bank', 'MTCH': 'Match Group',
    'MTD': 'Mettler-Toledo', 'MU': 'Micron Technology', 'NCLH': 'Norwegian Cruise Line Holdings',
    'NDAQ': 'Nasdaq, Inc.', 'NDSN': 'Nordson Corporation', 'NEE': 'NextEra Energy',
    'NEM': 'Newmont', 'NFLX': 'Netflix', 'NI': 'NiSource',
    'NKE': 'Nike, Inc.', 'NOC': 'Northrop Grumman', 'NOW': 'ServiceNow',
    'NRG': 'NRG Energy', 'NSC': 'Norfolk Southern', 'NTAP': 'NetApp',
    'NTRS': 'Northern Trust', 'NUE': 'Nucor', 'NVDA': 'NVIDIA',
    'NVR': 'NVR, Inc.', 'NWS': 'News Corp (Class B)', 'NWSA': 'News Corp (Class A)',
    'NXPI': 'NXP Semiconductors', 'O': 'Realty Income', 'ODFL': 'Old Dominion Freight Line',
    'OKE': 'ONEOK', 'OMC': 'Omnicom Group', 'ON': 'ON Semiconductor',
    'ORCL': 'Oracle Corporation', 'ORLY': 'O\'Reilly Auto', 'OTIS': 'Otis Worldwide',
    'OXY': 'Occidental Petroleum', 'PANW': 'Palo Alto Networks', 'PAYC': 'Paycom',
    'PAYX': 'Paychex', 'PCAR': 'PACCAR', 'PCG': 'PG&E Corporation',
    'PEG': 'Public Service Enterprise Group', 'PEP': 'PepsiCo', 'PFE': 'Pfizer',
    'PFG': 'Principal Financial Group', 'PG': 'Procter & Gamble', 'PGR': 'Progressive Corporation',
    'PH': 'Parker-Hannifin', 'PHM': 'PulteGroup', 'PKG': 'Packaging Corporation of America',
    'PLD': 'Prologis', 'PLTR': 'Palantir Technologies', 'PM': 'Philip Morris International',
    'PNC': 'PNC Financial Services', 'PNR': 'Pentair', 'PNW': 'Pinnacle West Capital',
    'PODD': 'Insulet', 'POOL': 'Pool Corporation', 'PPG': 'PPG Industries',
    'PPL': 'PPL Corporation', 'PRU': 'Prudential Financial', 'PSA': 'Public Storage',
    'PSX': 'Phillips 66', 'PTC': 'PTC Inc.', 'PWR': 'Quanta Services',
    'PYPL': 'PayPal', 'QCOM': 'Qualcomm', 'QRVO': 'Qorvo',
    'RCL': 'Royal Caribbean Group', 'REG': 'Regency Centers', 'REGN': 'Regeneron',
    'RF': 'Regions Financial Corporation', 'RJF': 'Raymond James', 'RL': 'Ralph Lauren Corporation',
    'RMD': 'ResMed', 'ROK': 'Rockwell Automation', 'ROL': 'Rollins, Inc.',
    'ROP': 'Roper Technologies', 'ROST': 'Ross Stores', 'RSG': 'Republic Services',
    'RTX': 'RTX Corporation', 'RVTY': 'Revvity', 'SBAC': 'SBA Communications',
    'SBUX': 'Starbucks', 'SCHW': 'Charles Schwab Corporation', 'SHW': 'Sherwin-Williams',
    'SJM': 'The J.M. Smucker Company', 'SLB': 'SLB (Schlumberger)', 'SMCI': 'Super Micro Computer',
    'SNA': 'Snap-on', 'SNPS': 'Synopsys', 'SO': 'Southern Company',
    'SOLV': 'Solventum', 'SPG': 'Simon Property Group', 'SPGI': 'S&P Global',
    'SRE': 'Sempra', 'STE': 'STERIS', 'STLD': 'Steel Dynamics',
    'STT': 'State Street Corporation', 'STX': 'Seagate Technology', 'STZ': 'Constellation Brands',
    'SW': 'Smurfit Westrock', 'SWK': 'Stanley Black & Decker', 'SWKS': 'Skyworks Solutions',
    'SYF': 'Synchrony Financial', 'SYK': 'Stryker Corporation', 'SYY': 'Sysco',
    'T': 'AT&T', 'TAP': 'Molson Coors', 'TDG': 'TransDigm Group',
    'TDY': 'Teledyne Technologies', 'TECH': 'Bio-Techne', 'TEL': 'TE Connectivity',
    'TER': 'Teradyne', 'TFC': 'Truist Financial', 'TFX': 'Teleflex',
    'TGT': 'Target Corporation', 'TJX': 'TJX Companies', 'TMO': 'Thermo Fisher Scientific',
    'TMUS': 'T-Mobile US', 'TPR': 'Tapestry, Inc.', 'TRGP': 'Targa Resources',
    'TRMB': 'Trimble Inc.', 'TROW': 'T. Rowe Price', 'TRV': 'The Travelers Companies',
    'TSCO': 'Tractor Supply Company', 'TSLA': 'Tesla, Inc.', 'TSN': 'Tyson Foods',
    'TT': 'Trane Technologies', 'TTWO': 'Take-Two Interactive', 'TXN': 'Texas Instruments',
    'TXT': 'Textron', 'TYL': 'Tyler Technologies', 'UAL': 'United Airlines',
    'UBER': 'Uber', 'UDR': 'UDR, Inc.', 'UHS': 'Universal Health Services',
    'ULTA': 'Ulta Beauty', 'UNH': 'UnitedHealth Group', 'UNP': 'Union Pacific',
    'UPS': 'United Parcel Service', 'URI': 'United Rentals', 'USB': 'U.S. Bancorp',
    'V': 'Visa Inc.', 'VICI': 'VICI Properties', 'VLO': 'Valero Energy',
    'VLTO': 'Veralto', 'VMC': 'Vulcan Materials Company', 'VRSK': 'Verisk',
    'VRSN': 'Verisign', 'VRTX': 'Vertex Pharmaceuticals', 'VST': 'Vistra',
    'VTR': 'Ventas', 'VTRS': 'Viatris', 'VZ': 'Verizon',
    'WAB': 'Wabtec', 'WAT': 'Waters Corporation', 'WBD': 'Warner Bros. Discovery',
    'WDC': 'Western Digital', 'WEC': 'WEC Energy Group', 'WELL': 'Welltower',
    'WFC': 'Wells Fargo', 'WM': 'Waste Management', 'WMB': 'Williams Companies',
    'WMT': 'Walmart', 'WRB': 'W. R. Berkley Corporation', 'WST': 'West Pharmaceutical Services',
    'WTW': 'Willis Towers Watson', 'WY': 'Weyerhaeuser', 'WYNN': 'Wynn Resorts',
    'XEL': 'Xcel Energy', 'XOM': 'ExxonMobil', 'XYL': 'Xylem Inc.',
    'YUM': 'Yum! Brands', 'ZBH': 'Zimmer Biomet', 'ZBRA': 'Zebra Technologies',
    'ZTS': 'Zoetis','Wall Street': 'Market General', 'Finance': 'Sector General', 
    'Trade': 'Economic Activity', 'Gold': 'Commodity: Gold', 'Oil': 'Commodity: Oil'
}


keywords_regex = '|'.join([re.escape(str(t)) for t in ticker_list])

merged_df['finance_flag'] = merged_df['Text'].str.contains(keywords_regex, case = False).astype(int)

financial = merged_df[merged_df['finance_flag'] ==1]

financial.columns.to_list()


#making it long


id_vars = ['Date', 'Text']


id_vars = [
    'Date',
    'Text',
    'exception_flag',
    'Trimmed_Text',
    'finance_flag'
]

value_vars = [
    'vader_polarity',
    'mean_polarity',
    'vader_polarity_classification',
    'Fin_Sent_Label',
    'Fin_Sent_Score',
    'News_Headlines_Label',
    'News_Headlines_Score'
]

long_df = financial.melt(
    id_vars=id_vars,
    value_vars=value_vars,
    var_name='sentiment_type',
    value_name='sentiment'
)


long_df.columns.to_list()

long_df = long_df.rename(columns={'sentiment': 'value'})

long_df.to_parquet("FinalPolarityData.parquet")

long_df['value'] = long_df['value'].astype(str)
long_df.to_parquet("FinalPolarityData.parquet")