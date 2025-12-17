import yfinance as yf
import pandas as pd
import time
from datetime import datetime

def fetch_nasdaq_tickers():
    """
    Fetch top 1000 NASDAQ tickers with sector information and current market data.
    This uses a combination of sources to get comprehensive ticker data.
    """
    print("Fetching NASDAQ ticker list...")

    tickers_list = []

    # Get NASDAQ-100 from Wikipedia
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        tables = pd.read_html(url)
        nasdaq_100 = tables[4]
        tickers_list.extend(nasdaq_100['Ticker'].tolist())
        print(f"Found {len(tickers_list)} tickers from NASDAQ-100")
    except Exception as e:
        print(f"Could not fetch NASDAQ-100 from Wikipedia: {e}")

    # Comprehensive list of top NASDAQ stocks by sector
    additional_tickers = [
        # Technology - Major
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'QCOM',
        'AVGO', 'CSCO', 'ADBE', 'NFLX', 'CRM', 'ORCL', 'ACN', 'TXN', 'IBM', 'NOW',
        'INTU', 'AMAT', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'ADSK', 'MCHP', 'FTNT',
        'PANW', 'WDAY', 'TEAM', 'DDOG', 'CRWD', 'ZS', 'NET', 'SNOW', 'OKTA', 'ANSS',
        'TTWO', 'EA', 'ATVI', 'VEEV', 'VRSN', 'FFIV', 'JNPR', 'AKAM', 'NTAP', 'RNG',
        'PTC', 'GDDY', 'SMCI', 'DELL', 'HPQ', 'NTCT', 'CWAN', 'JAMF', 'IOT', 'APPN',

        # Technology - Software & Cloud
        'ESTC', 'MDB', 'CFLT', 'GTLB', 'S', 'BILL', 'PCTY', 'ZI', 'DOCN', 'FSLY',
        'FROG', 'HUBS', 'ASAN', 'MNDY', 'ZUO', 'RNG', 'BOX', 'BLKB', 'SMAR', 'BRZE',
        'ALRM', 'NEWR', 'COUP', 'VRNS', 'QTWO', 'BL', 'SSNC', 'BR', 'FICO', 'GWRE',
        'AZPN', 'CTXS', 'CSGP', 'MANH', 'NSIT', 'QLYS', 'TENB', 'RPD', 'CVLT', 'APPF',

        # Consumer Discretionary
        'AMZN', 'BKNG', 'SBUX', 'MAR', 'CMG', 'ORLY', 'ROST', 'LULU', 'ABNB', 'EBAY',
        'MCD', 'YUM', 'CPRT', 'CTAS', 'FAST', 'DLTR', 'ODFL', 'ULTA', 'POOL', 'AZO',
        'EXPE', 'BFAM', 'SEAS', 'SIX', 'HLT', 'MGM', 'WYNN', 'LVS', 'CZR', 'GRMN',
        'BBY', 'BBWI', 'BBBY', 'URBN', 'ANF', 'GPS', 'EXPR', 'AEO', 'CATO', 'CROX',
        'BOOT', 'DDS', 'GCO', 'SCVL', 'LE', 'HIMS', 'PLAY', 'TXRH', 'BLMN', 'CAKE',
        'DIN', 'BJRI', 'DENN', 'FRGI', 'KRUS', 'LOCO', 'PZZA', 'QSR', 'WING', 'BWLD',

        # Communication Services
        'GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'TMUS', 'CHTR', 'EA', 'TTWO',
        'MTCH', 'SNAP', 'PINS', 'SPOT', 'RBLX', 'ZM', 'DOCU', 'TWLO', 'U', 'LBRDK',
        'LBTYK', 'LBTYA', 'FOXA', 'FOX', 'DISCA', 'DISCK', 'VIAC', 'VIACA', 'NWSA',
        'SHEN', 'YELP', 'QUOT', 'CARS', 'TGNA', 'NXST', 'LEG', 'MANU', 'MSG', 'MSGN',

        # Healthcare / Biotech - Major
        'AMGN', 'GILD', 'BIIB', 'VRTX', 'REGN', 'ILMN', 'MRNA', 'ALGN', 'IDXX', 'DXCM',
        'ISRG', 'INCY', 'BMRN', 'EXAS', 'HOLX', 'TECH', 'PODD', 'ICLR', 'MEDP', 'XRAY',
        'NBIX', 'UTHR', 'IONS', 'RARE', 'LEGN', 'FOLD', 'RVMD', 'ARWR', 'HALO', 'SRPT',
        'JAZZ', 'ALKS', 'ACAD', 'CORT', 'HZNP', 'IMGN', 'AXSM', 'RPRX', 'ITCI', 'TBPH',
        'CERE', 'DNLI', 'EHTH', 'HCAT', 'LHCG', 'PDCO', 'POOL', 'QDEL', 'RDNT', 'SDGR',

        # Biotech - Gene Therapy & CRISPR
        'FATE', 'CRSP', 'EDIT', 'NTLA', 'BEAM', 'BLUE', 'VCYT', 'PACB', 'CDNA', 'ILMN',
        'PTGX', 'SGMO', 'RXRX', 'SANA', 'RCKT', 'ALNY', 'AGIO', 'ARCT', 'ARVN', 'ATRA',
        'BCRX', 'BHVN', 'BPMC', 'CARA', 'CCCC', 'CLDX', 'CMPS', 'COCP', 'CTMX', 'CVAC',
        'DAWN', 'DCPH', 'DRNA', 'DVAX', 'DYAI', 'EIGR', 'EOLS', 'EPIX', 'ERAS', 'FENC',

        # Biotech - Oncology & Immunotherapy
        'EXEL', 'FGEN', 'GTHX', 'IGMS', 'IMAB', 'IMDZ', 'IMMP', 'IMRN', 'IMTX', 'IMVT',
        'INSM', 'INVA', 'IOVA', 'ITOS', 'JANX', 'KPTI', 'KROS', 'KRTX', 'KYMR', 'LBPH',
        'LEGN', 'LENZ', 'LIAN', 'LNTH', 'LONC', 'LPTX', 'LRMR', 'LTRN', 'LXRX', 'LYEL',
        'MDGL', 'MGNX', 'MIRM', 'MRUS', 'MRSN', 'MRTX', 'MYGN', 'NBTX', 'NKTR', 'NRIX',

        # Consumer Staples
        'COST', 'PEP', 'MDLZ', 'MNST', 'KHC', 'KDP', 'COKE', 'FIZZ', 'CELH', 'CALM',
        'JBSS', 'LANC', 'LWAY', 'SAFM', 'SMPL', 'UNFI', 'USFD', 'WDFC', 'BGS', 'CENT',
        'CHEF', 'COKE', 'CVGW', 'EPC', 'FDP', 'HAIN', 'INGR', 'JJSF', 'MGPI', 'SENEA',
        'SJM', 'SPTN', 'THS', 'TSN', 'VITL', 'POST', 'PFGC', 'NOMD', 'FRPT', 'BYND',

        # Industrials
        'HON', 'UPS', 'LMT', 'RTX', 'CAT', 'DE', 'BA', 'GD', 'NOC', 'LHX',
        'PCAR', 'VRSK', 'PAYX', 'FAST', 'CTAS', 'ODFL', 'CHRW', 'JBHT', 'EXPD', 'XPO',
        'ALSN', 'ATKR', 'AYI', 'BLDR', 'BWA', 'CARR', 'CMI', 'CR', 'CSL', 'CSWI',
        'DY', 'ECHO', 'EME', 'ESAB', 'FLS', 'GGG', 'GMS', 'GVA', 'GWW', 'HUBG',
        'IEX', 'ITT', 'JBLU', 'JOBY', 'KEX', 'KNX', 'LSTR', 'MAS', 'MATX', 'MLI',
        'MSM', 'NPO', 'OSK', 'PATK', 'PLUG', 'PNR', 'R', 'RBC', 'RXO', 'SAIA',

        # Financials / FinTech
        'PYPL', 'COIN', 'MELI', 'FISV', 'LPLA', 'HOOD', 'SOFI', 'AFRM', 'UPST',
        'STNE', 'PAGS', 'NU', 'TOST', 'GPN', 'LC', 'TREE', 'LPRO', 'OPFI', 'BTBT',
        'ACIW', 'AUB', 'BANC', 'BANR', 'BMRC', 'BPOP', 'BRKL', 'BY', 'CADE', 'CASH',
        'CATY', 'CBSH', 'CBNK', 'CFFN', 'CFG', 'COLB', 'COOP', 'CVBF', 'DCOM', 'EGBN',
        'EQBK', 'EWBC', 'FCNCA', 'FELE', 'FFBC', 'FFIN', 'FIBK', 'FNB', 'FRME', 'FULT',

        # Energy & Clean Energy
        'ENPH', 'SEDG', 'PLUG', 'FCEL', 'BE', 'RUN', 'ARRY', 'SHLS', 'CLNE', 'NOVA',
        'CSIQ', 'DQ', 'JKS', 'FSLR', 'SPWR', 'MAXN', 'SLDP', 'QS', 'BLNK', 'CHPT',
        'EVGO', 'PTRA', 'NKLA', 'LEV', 'STEM', 'OUST', 'LAZR', 'VLTA', 'FREYR', 'MP',
        'REE', 'ARVL', 'ENVX', 'ABML', 'AMPX', 'BEEM', 'CWEN', 'FLUX', 'HDSN', 'NEP',

        # Materials
        'APD', 'ECL', 'SHW', 'DD', 'NEM', 'FCX', 'ALB', 'CE', 'CTVA', 'ASH',
        'AXTA', 'BCPC', 'CBT', 'CC', 'CF', 'CMC', 'CRS', 'DNOW', 'DOV', 'EMN',
        'FUL', 'HAYN', 'HUN', 'IP', 'KWR', 'LTHM', 'MLM', 'MOS', 'NUE', 'OLN',
        'PKG', 'PPG', 'RPM', 'SEE', 'SXT', 'USLM', 'VMC', 'WLK', 'X', 'XYL',

        # Real Estate
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'DLR', 'SBAC', 'WELL', 'AVB', 'EQR',
        'ARE', 'BXP', 'CPT', 'CUBE', 'DEI', 'DOC', 'DRE', 'EGP', 'ELS', 'EPR',
        'ESS', 'EXR', 'FR', 'GLPI', 'HIW', 'HST', 'INVH', 'IRM', 'KIM', 'KRC',
        'LSI', 'MAC', 'MAA', 'MPW', 'NNN', 'NSA', 'O', 'OHI', 'PEAK', 'PK',

        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'EXC', 'AEP', 'SRE', 'XEL', 'WEC', 'ED',
        'AEE', 'AES', 'AGR', 'ALE', 'AQUA', 'ATO', 'AVA', 'AWK', 'AWR', 'BKH',
        'CMS', 'CNP', 'CPK', 'CWT', 'DTE', 'EVRG', 'FE', 'HURC', 'LNT', 'MDU',
        'NI', 'NJR', 'NWE', 'NWN', 'OGE', 'ORA', 'OTTR', 'PNW', 'POR', 'PPL',

        # EV and Automotive
        'TSLA', 'LCID', 'RIVN', 'NIO', 'XPEV', 'LI', 'GOEV', 'WKHS', 'HYLN', 'FSR',
        'RIDE', 'NKLA', 'ARVL', 'LEV', 'GGPI', 'MULN', 'SOLO', 'AYRO', 'BLNK', 'CHPT',
        'PSNY', 'QS', 'VLTA', 'NUVVE', 'ELMS', 'INDI', 'ORGN', 'REE', 'HYZN', 'GEV',

        # Semiconductors
        'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'MU', 'AMAT', 'LRCX', 'KLAC',
        'MRVL', 'MCHP', 'ADI', 'NXPI', 'SWKS', 'QRVO', 'MPWR', 'ON', 'TER', 'ENTG',
        'ALGM', 'AMKR', 'AOSL', 'AMBA', 'AMPH', 'ATHA', 'AXTI', 'BRKS', 'CRUS', 'DIOD',
        'FORM', 'ICHR', 'INDI', 'INTC', 'IRDM', 'KLIC', 'LSCC', 'MKSI', 'MTSI', 'NVMI',
        'NVTS', 'PLAB', 'POWI', 'QCOM', 'RMBS', 'SITM', 'SMTC', 'UCTT', 'WOLF', 'XLNX',

        # Cloud/SaaS Additional
        'APPN', 'ASAN', 'BASE', 'BLKB', 'BOX', 'BZ', 'CWAN', 'DT', 'DV', 'FIVN',
        'FRSH', 'FROG', 'HUBS', 'JAMF', 'MNDY', 'PATH', 'PD', 'PLAN', 'QLYS', 'RNG',
        'SMAR', 'TENB', 'TDC', 'VEEV', 'WK', 'WDAY', 'WIX', 'YEXT', 'ZEN', 'ZUO',

        # E-commerce
        'AMZN', 'SHOP', 'MELI', 'EBAY', 'ETSY', 'W', 'CHWY', 'CVNA', 'GRPN', 'RVLV',
        'APRN', 'BIGC', 'CRWD', 'DASH', 'FTCH', 'JMIA', 'LFST', 'OZON', 'PRTS', 'QUOTW',
        'REAL', 'RSKD', 'RXDX', 'SEED', 'STNE', 'VIPS', 'VZIO', 'WISH', 'WOOF', 'WSC',

        # Gaming
        'EA', 'TTWO', 'RBLX', 'U', 'PLTK', 'DKNG', 'PENN', 'RSI', 'BALY', 'CHDN',
        'CZR', 'DKNG', 'EVRI', 'FLUT', 'FUBO', 'GENI', 'IGT', 'LNW', 'LUCK', 'MGM',
        'PENN', 'PLBY', 'RSI', 'SKLZ', 'WYNN', 'ZNGA', 'AKAM', 'GLUU', 'GNOG', 'GMBL',

        # Transportation & Logistics
        'UBER', 'LYFT', 'DASH', 'WING', 'BLBD', 'JBLU', 'ALK', 'SAVE', 'HA',
        'ARCB', 'CVLG', 'ECHO', 'EXPD', 'FWRD', 'GNK', 'JBHT', 'KEX', 'LSTR', 'MATX',
        'MRTN', 'ODFL', 'SAIA', 'SNDR', 'TRN', 'UPS', 'WERN', 'XPO', 'YELL', 'ZIM',

        # Retail
        'COST', 'WMT', 'HD', 'LOW', 'TGT', 'TJX', 'ROST', 'DLTR', 'DG', 'FIVE',
        'ACI', 'ANF', 'BBBY', 'BBY', 'BIG', 'BKE', 'BOOT', 'BURL', 'CAL', 'CASY',
        'CHWY', 'CONN', 'CRI', 'CROX', 'DDS', 'EXPR', 'FL', 'GES', 'GPI', 'GPS',
        'HIBB', 'KSS', 'LE', 'M', 'OLLI', 'OSTK', 'PETM', 'PLCE', 'PSMT', 'RVLV',

        # Media & Entertainment
        'NFLX', 'DIS', 'CMCSA', 'WBD', 'ROKU', 'FUBO', 'MSGS', 'IMAX', 'PARA',
        'AMC', 'CNK', 'DISH', 'EDR', 'FOXA', 'GTN', 'IPG', 'LGF.A', 'LILAK', 'LLYVA',
        'MSGM', 'NWSA', 'OMC', 'QNST', 'SCHL', 'STGW', 'TGNA', 'THRYV', 'TDS', 'VMEO',

        # Cybersecurity
        'PANW', 'CRWD', 'ZS', 'FTNT', 'OKTA', 'CYBR', 'TENB', 'RPD', 'QLYS', 'CHKP',
        'SAIL', 'S', 'VRNS', 'CWAN', 'RBRK', 'PING', 'AVDX', 'VRRM', 'SCWX', 'NLOK',
        'PFPT', 'FEYE', 'PANW', 'FTNT', 'AKAM', 'JNPR', 'CSCO', 'FFIV', 'NTCT', 'ATEN',

        # Artificial Intelligence & Robotics
        'NVDA', 'AMD', 'GOOGL', 'MSFT', 'META', 'AI', 'PLTR', 'PATH', 'SNOW', 'NET',
        'BBAI', 'BIGC', 'CDLX', 'GDRX', 'RSKD', 'SOUN', 'SSYS', 'SWI', 'VRT', 'VZIO',
        'AMBA', 'CEVA', 'DM', 'IRBT', 'ISRG', 'JOBY', 'KTOS', 'LUNR', 'NNDM', 'PRLB',
        'PRNT', 'SNPS', 'TER', 'VERI', 'XMTR', 'XXII', 'AVAV', 'BLDE', 'SPIR', 'UAVS',

        # Payments & Processing
        'PYPL', 'AFRM', 'SOFI', 'UPST', 'STNE', 'PAGS', 'NU', 'TOST', 'GPN',
        'AXP', 'CPAY', 'DLO', 'EEFT', 'EVTC', 'FARO', 'FISV', 'FLT', 'GPN', 'GWRE',
        'JXN', 'NCR', 'PAR', 'PAYO', 'PRFT', 'PTC', 'QTWO', 'REPAY', 'TPAY', 'VCTR',

        # Medical Devices
        'ISRG', 'ALGN', 'DXCM', 'PODD', 'HOLX', 'ICUI', 'NVST', 'NVCR', 'IRTC', 'TNDM',
        'ATRC', 'AXNX', 'BDX', 'BSX', 'EW', 'GMED', 'GKOS', 'IART', 'INSP', 'LIVN',
        'LMAT', 'MDT', 'MMSI', 'NEOG', 'NVCR', 'OMCL', 'OFIX', 'RXST', 'STE', 'STAA',
        'SYK', 'TFX', 'TDOC', 'VCYT', 'VREX', 'VSTM', 'WST', 'XRAY', 'ZBH', 'ZIMV',

        # Food Delivery & Gig Economy
        'DASH', 'UBER', 'ABNB', 'LYFT', 'CHEF', 'BMBL', 'MTCH', 'CVNA', 'GRUB',
        'BKNG', 'EXPE', 'TRIP', 'MMYT', 'TZOO', 'DESP', 'PCLN', 'OWLT', 'TRVG', 'AWAY',
        'TCOM', 'WFRD', 'WYND', 'RDFN', 'OPEN', 'COMP', 'RMAX', 'HOUS', 'EXPI', 'ZILG',

        # 5G and Telecom Equipment
        'QCOM', 'AVGO', 'MRVL', 'SWKS', 'QRVO', 'COMM', 'SLAB', 'CIEN', 'LITE', 'AMBA',
        'ACIA', 'ADTN', 'CALX', 'COMM', 'CSCO', 'EGHT', 'EXTR', 'FNSR', 'IDCC', 'INFN',
        'JNPR', 'LUMN', 'MTSI', 'NOK', 'VIAV', 'ATUS', 'CABO', 'LBRDA', 'LBRDK', 'TMUS',

        # Cannabis
        'TLRY', 'CGC', 'CRON', 'ACB', 'SNDL', 'OGI', 'VFF', 'GRWG', 'HEXO', 'APHA',
        'CRON', 'MJ', 'IIPR', 'TCNNF', 'VRNOF', 'CURLF', 'GTBIF', 'CRLBF', 'TRUL', 'HRVSF',
        'KERN', 'SMG', 'GPOR', 'TPCO', 'JUSHF', 'GRAMF', 'PLNHF', 'MMNFF', 'MSOS', 'YOLO',

        # Solar Additional
        'MAXN', 'NOVA', 'SOL', 'SHLS', 'ASTI', 'CSIQ', 'DQ', 'ENPH', 'FSLR', 'JKS',
        'RUN', 'SEDG', 'SPWR', 'TAN', 'ARRY', 'CSUN', 'PECK', 'REGI', 'VSLR', 'WATT',

        # Streaming & Audio
        'NFLX', 'ROKU', 'FUBO', 'SPOT', 'SIRI', 'PTON', 'SONO', 'GPRO', 'KOSS',
        'VUZI', 'GNUS', 'HEAR', 'KOSS', 'LTRX', 'SIRI', 'UEIC', 'VEON', 'VG', 'VNET',
        'WBD', 'DIS', 'PARA', 'NFLX', 'CMCSA', 'DISH', 'ROKU', 'TTD', 'MGNI', 'APPS',

        # Workout & Fitness
        'PTON', 'LULU', 'NKE', 'PLNT', 'YETI', 'VFC', 'UAA', 'COLM', 'DECK', 'ELY',
        'MODG', 'GIII', 'SKX', 'CROX', 'WWW', 'ZUMZ', 'BGFV', 'LAKE', 'GPI', 'HIBB',
        'FL', 'DKS', 'ASO', 'BOOT', 'CAL', 'SHOO', 'WTRG', 'PRTY', 'EXPR', 'TLRD',

        # Data Centers & Infrastructure
        'DLR', 'EQIX', 'CONE', 'QTS', 'SBAC', 'AMT', 'CCI', 'UNIT', 'SAFE', 'CUTR',
        'LAND', 'DRH', 'CLNC', 'CIO', 'DFH', 'REXR', 'STAG', 'VNO', 'BNL', 'FCPT',

        # Insurance Tech
        'LMND', 'ROOT', 'METC', 'OSCR', 'TRUP', 'KINS', 'OPEN', 'HIPO', 'CLVR', 'BTRS',
        'BWIN', 'CTIC', 'DHIL', 'EIG', 'ERIE', 'ESGR', 'GSHD', 'HCI', 'JRVR', 'KNSL',

        # Construction & Infrastructure
        'BLDR', 'BECN', 'CVCO', 'FAST', 'FIX', 'GNRC', 'GVA', 'HBB', 'IBP', 'JELD',
        'KBH', 'LEN', 'MHK', 'NX', 'OC', 'POOL', 'SSD', 'TMHC', 'TOL', 'TPH',
        'TREX', 'UFPI', 'VMC', 'WCC', 'WFCF', 'WY', 'AWI', 'AZEK', 'BERY', 'BZH'
    ]

    # Add unique tickers only
    for ticker in additional_tickers:
        if ticker not in tickers_list:
            tickers_list.append(ticker)

    # Limit to ~1000 tickers
    tickers_list = tickers_list[:1000]

    print(f"Total tickers to process: {len(tickers_list)}")

    # Prepare data storage
    ticker_data = []
    failed_tickers = []

    for i, ticker_symbol in enumerate(tickers_list, 1):
        try:
            print(f"Processing {i}/{len(tickers_list)}: {ticker_symbol}", end='')

            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            hist = ticker.history(period='5d')

            if len(hist) == 0:
                print(" - No price data, skipping")
                failed_tickers.append(ticker_symbol)
                continue

            last_price = hist['Close'].iloc[-1]

            ticker_data.append({
                'Ticker': ticker_symbol,
                'Company Name': info.get('longName', info.get('shortName', 'N/A')),
                'Sector': info.get('sector', 'N/A'),
                'Industry': info.get('industry', 'N/A'),
                'Market Cap': info.get('marketCap', 0),
                'Last Price': round(last_price, 2),
                'Volume': hist['Volume'].iloc[-1],
                '52 Week High': info.get('fiftyTwoWeekHigh', 0),
                '52 Week Low': info.get('fiftyTwoWeekLow', 0),
                'Average Volume': info.get('averageVolume', 0),
                'P/E Ratio': info.get('trailingPE', 0),
                'Dividend Yield': info.get('dividendYield', 0),
                'Beta': info.get('beta', 0),
                'Last Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            print(f" - ${last_price:.2f}")

            # Rate limiting to avoid getting blocked
            if i % 10 == 0:
                time.sleep(1)

        except Exception as e:
            print(f" - Error: {str(e)[:50]}")
            failed_tickers.append(ticker_symbol)
            continue

    # Create DataFrame
    df = pd.DataFrame(ticker_data)

    # Sort by Market Cap (largest first)
    df = df.sort_values('Market Cap', ascending=False)

    # Save to CSV
    csv_filename = 'nasdaq_tickers.csv'
    df.to_csv(csv_filename, index=False)

    print(f"\n{'='*60}")
    print(f"Successfully saved {len(df)} tickers to {csv_filename}")
    print(f"Failed to fetch: {len(failed_tickers)} tickers")
    print(f"\nSector breakdown:")
    print(df['Sector'].value_counts())
    print(f"{'='*60}")

    return df

if __name__ == '__main__':
    df = fetch_nasdaq_tickers()
    print(f"\nFirst few entries:")
    print(df.head(10)[['Ticker', 'Company Name', 'Sector', 'Last Price', 'Market Cap']])
