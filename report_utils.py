import math
import pandas as pd

# Due to the extensive national catalogs that are going to be translated, 
# I decided to use multithreading to complete the translation to run faster
from concurrent.futures import ThreadPoolExecutor, as_completed

import os

# Importing a progress bar due to how big the catalog dataset are
from tqdm.notebook import tqdm

# Unfortunately, as the catalogs are in Spanish, we need to translate the data dictionary to identify the columns
from deep_translator import GoogleTranslator

import asyncio
import nest_asyncio
import aiohttp
nest_asyncio.apply()
import json

import time


# Locales for the Google Translator
english = 'en'
spanish = 'es'
german = 'de'

# Setting up the main directories
main_folder = os.getcwd()
data_folder = os.path.join(main_folder, 'data')
generated_data_folder = os.path.join(main_folder, 'generated_data')

translator = GoogleTranslator(source = spanish, target = english) 

request_tracking = {
    'successful': {},
    'failed': {},
    'rate_limited': [],
    'total_attempted': 0,
    'start_time': time.time(),
    'last_request_time': 0
}

paraguay_tender_contract_column = 'tender_publications_lastcontract'
paraguay_tender_contracts_list = []

paraguay_procurement_data = pd.read_csv(os.path.join(data_folder, 'dfid2_py_210715_csv.csv'))

paraguay_tender_publications_list = paraguay_procurement_data[paraguay_tender_contract_column].to_list()

for tender in tqdm(paraguay_tender_publications_list, desc='Processing Tender Contracts'):
    if 'contract' in tender:
        paraguay_tender_contracts_list.append(tender)

# Here are the functions that are used to complete the Spanish to English Translations (if the file doesn't exist)
def translate_column_batch(col_data, batch_size=50): 
    """ Translate a single column using batch translation. Returns: (column_name, translated_values, indices) """ 
    col_name, series = col_data 
    non_null_mask = series.notna() 
    original_values = series.loc[non_null_mask].astype(str) 
    n = len(original_values) 
    if n == 0: 
        return col_name, [], original_values.index 
    translated_values = [] 
    num_batches = math.ceil(n / batch_size) 

    for i in tqdm(range(num_batches), desc=f"Translating {col_name}", leave=False):
        start = i * batch_size
        end = min((i + 1) * batch_size, n)
        batch = list(original_values.iloc[start:end])
        try:
            translated_batch = translator.translate_batch(batch)
        except Exception:
            translated_batch = []
            for text in tqdm(batch, desc = f"Fallback translation", leave=False):
                try:
                    translated_batch.append(translator.translate(text))
                except:
                    translated_batch.append(text) 
        translated_values.extend(translated_batch)
    return col_name, translated_values, original_values.index

def translate_dataframe_threading(df, columns_to_translate, max_workers = None, batch_size=50):
    df_translated = df.copy()
    col_data_list = [(col, df[col]) for col in columns_to_translate]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(translate_column_batch, col_data, batch_size): col_data[0] for col_data in col_data_list}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Translating columns"):
            col_name, translated_values, indices = f.result()
            df_translated.loc[indices, col_name] = translated_values
    return df_translated

# This function encapsulates the above functions so that only the arguments need to be passed for each country and their catalogs
def translate_country_procurement_catalog(catalog_df, file_name, file_type, drop_columns = None):
    translated_catalog = catalog_df.copy()

    translation_required = False
    for i in os.listdir(generated_data_folder):
        if f'translated_{file_name}.{file_type}' not in os.listdir(generated_data_folder):
            translation_required = True        
    
    if translation_required:
        print('File Not Found, Running Translation')
        if drop_columns is not None:
            translate_columns = translated_catalog.columns.drop(drop_columns)
        else:
            translate_columns = translated_catalog.columns

        translated_catalog = translate_dataframe_threading(
            catalog_df, translate_columns, max_workers=4, batch_size=50
        )

        translated_catalog.to_csv(os.path.join(generated_data_folder, f'translated_{file_name}.{file_type}'), index = False)
        
    else:
        print('File Found. Loading Dataset')
        translated_catalog = pd.read_csv(os.path.join(generated_data_folder, f'translated_{file_name}.{file_type}'))
    return translated_catalog

# This is used to check what the columns of interest have
def print_columns_check(dataset, country):
    print(f'Procedure Types for Tenders in {country}:', dataset['tender_nationalproceduretype'].value_counts())
    print('')
    print(f'Number of Recorded Bids in {country}:', dataset['tender_recordedbidscount'].value_counts())
    print('')
    print(f'Number of Buyers in {country}: ', dataset['buyer_name'].nunique())
    print('')
    print(f'Number of Bidders in {country}: ', dataset['bidder_name'].nunique())
    print('')
    print(f'Tender Publication Call Dates in {country}: ', dataset['tender_publications_firstcallfor'].value_counts())
    print('')
    print(f'Tender Bid Deadlines in {country}: ', dataset['tender_biddeadline'].value_counts())
    print('')
    print(f'Tender Award Dates in {country}: ', dataset['tender_contractsignaturedate'].value_counts())

async def fetch_with_rate_limiting(session, url, sem, url_index, tracking_dict):
    """Fetch with aggressive rate limiting"""
    async with sem:
        try:            
            # Wait at least 2.5 seconds between requests
            await asyncio.sleep(2.5)
            
            tracking_dict['last_request_time'] = time.time()
            tracking_dict['total_attempted'] += 1
            
            async with session.get(url, timeout=120) as response:
                response_text = await response.text()

                # Check for timeout error
                timeout_error = False
                
                # Check for rate limiting (more comprehensive)
                rate_limit_indicators = [
                    'límite de peticiones',
                    'rate limit',
                    'too many requests',
                    'rate exceeded',
                    'quota exceeded'
                ]
                
                is_rate_limited = (
                    response.status == 429 or
                    response.status == 503 or
                    any(indicator in response_text.lower() for indicator in rate_limit_indicators)
                )
                
                if is_rate_limited:
                    tracking_dict['rate_limited'].append(url)
                    print(f'\n⚠️  Rate limit hit for URL {url_index}: {url[:50]}...')
                    print('🐌 Backing off for 5 seconds before retrying...')
                    
                    await asyncio.sleep(5.0)

                    async with session.get(url, timeout=120) as retry_response:
                        retry_response_text = await retry_response.text()
                        if retry_response.status == 200:
                            try:
                                parsed_json = json.loads(retry_response_text)
                                if 'releases' in parsed_json and len(parsed_json['releases']) > 0:
                                    tracking_dict['successful'][url] = {
                                        'response': retry_response_text,
                                        'url_index': url_index,
                                        'status_code': retry_response.status,
                                        'timestamp': time.time(),
                                        'response_length': len(retry_response_text)
                                    }
                                    print(f'✅ Success on retry for URL {url_index}\n')
                                    return retry_response_text
                            except json.JSONDecodeError:
                                pass
                        tracking_dict['failed'][url] = {
                            'error_type': 'rate_limited_retry_failed',
                            'status_code': retry_response.status,
                            'message': 'Rate limit exceeded on retry',
                            'url_index': url_index,
                            'timestamp': time.time(),
                            'response_preview': retry_response_text[:200]
                        }
                        print(f'❌ Retry failed for URL {url_index}\n')
                    return None
                
                # Check for other HTTP errors
                elif response.status != 200:
                    tracking_dict['failed'][url] = {
                        'error_type': 'http_error',
                        'status_code': response.status,
                        'message': f'HTTP {response.status}',
                        'url_index': url_index,
                        'timestamp': time.time()
                    }
                    print(f'❌ HTTP {response.status} for URL {url_index}')
                    return None
                
                # Validate JSON response
                try:
                    parsed_json = json.loads(response_text)
                    
                    # Check if response has expected structure
                    if 'releases' in parsed_json and len(parsed_json['releases']) > 0:
                        tracking_dict['successful'][url] = {
                            'response': response_text,
                            'url_index': url_index,
                            'status_code': response.status,
                            'timestamp': time.time(),
                            'response_length': len(response_text)
                        }
                        print(f'✅ Success for URL {url_index}')
                        return response_text
                    else:
                        tracking_dict['failed'][url] = {
                            'error_type': 'invalid_structure',
                            'status_code': response.status,
                            'message': 'Response missing expected structure',
                            'url_index': url_index,
                            'timestamp': time.time()
                        }
                        print(f'Failed Request for URL {url_index}')
                        return None
                        
                except json.JSONDecodeError:
                    tracking_dict['failed'][url] = {
                        'error_type': 'invalid_json',
                        'status_code': response.status,
                        'message': 'Response is not valid JSON',
                        'url_index': url_index,
                        'timestamp': time.time(),
                        'response_preview': response_text[:200]
                    }
                    print(f'⚠️  Invalid JSON for URL {url_index}')
                    return None
                    
        except asyncio.TimeoutError:
            timeout_error = True
            if timeout_error:
                print(f'⏰ Timeout for URL {url_index}. Retrying...')
                await asyncio.sleep(3.0)  # Wait before retrying

                async with session.get(url, timeout=120) as retry_response:
                    retry_response_text = await retry_response.text()
                    if retry_response.status == 200:
                        try:
                            parsed_json = json.loads(retry_response_text)
                            if 'releases' in parsed_json and len(parsed_json['releases']) > 0:
                                tracking_dict['successful'][url] = {
                                    'response': retry_response_text,
                                    'url_index': url_index,
                                    'status_code': retry_response.status,
                                    'timestamp': time.time(),
                                    'response_length': len(retry_response_text)
                                }
                                print(f'✅ Success on retry after timeout for URL {url_index}\n')
                                return retry_response_text
                        except json.JSONDecodeError:
                            pass
                    tracking_dict['failed'][url] = {
                        'error_type': 'timeout_retry_failed',
                        'status_code': retry_response.status,
                        'message': 'Timeout occurred and retry failed',
                        'url_index': url_index,
                        'timestamp': time.time(),
                        'response_preview': retry_response_text[:200]
                    }
                    print(f'❌ Retry after timeout failed for URL {url_index}\n')
                return None

            tracking_dict['failed'][url] = {
                'error_type': 'timeout',
                'status_code': None,
                'message': 'Request timeout (60s)',
                'url_index': url_index,
                'timestamp': time.time()
            }
            return None
            
        except Exception as e:
            tracking_dict['failed'][url] = {
                'error_type': 'exception',
                'status_code': None,
                'message': str(e),
                'url_index': url_index,
                'timestamp': time.time()
            }
            print(f'💥 Error for URL {url_index}: {str(e)[:50]}...')
            return None

async def get_tender_json_with_rate_limiting():
    # Set semaphore to 1 for maximum rate limit protection
    sem = asyncio.Semaphore(1)
    
    # Use custom headers to be more polite
    headers = {
        'User-Agent': 'Research/1.0 (Academic Study; aidan@example.com)',
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip, deflate'
    }
    
    # Use connection pooling for efficiency
    connector = aiohttp.TCPConnector(
        limit=1,  # Only 1 connection total
        limit_per_host=1,  # Only 1 connection per host
        ttl_dns_cache=300,
        use_dns_cache=True
    )
    
    timeout = aiohttp.ClientTimeout(total=60, connect=30)
    
    async with aiohttp.ClientSession(
        headers=headers,
        connector=connector,
        timeout=timeout
    ) as session:
        
        print(f'🚀 Starting to fetch {len(paraguay_tender_contracts_list)} contracts with aggressive rate limiting...')
        print(f'📊 Estimated time: {len(paraguay_tender_contracts_list)} seconds minimum')
        
        results = []
        
        # Process URLs one by one to maintain order and control
        for i, tender_url in enumerate(tqdm(paraguay_tender_contracts_list, desc='Fetching Contracts')):
            try:
                result = await fetch_with_rate_limiting(
                    session, tender_url, sem, i, request_tracking
                )
                results.append(result)
                
                # Progress reporting every 50 requests
                if (i + 1) % 50 == 0:
                    successful = len(request_tracking['successful'])
                    failed = len(request_tracking['failed'])
                    rate_limited = len(request_tracking['rate_limited'])
                    
                    print(f"\n📈 Progress: {i + 1}/{len(paraguay_tender_contracts_list)}")
                    print(f"✅ Success: {successful} | ❌ Failed: {failed} | ⚠️  Rate Limited: {rate_limited}")
                    print(f"📊 Success Rate: {successful/(i+1)*100:.1f}%")
                    
                    # If too many rate limits, slow down even more
                    if rate_limited > successful * 0.1:  # More than 10% rate limited
                        print("🐌 High rate limiting detected - adding extra delay...")
                        await asyncio.sleep(3.0)
                        
            except Exception as e:
                print(f'💥 Unexpected error for URL {i}: {e}')
                results.append(None)
    
    return results

def find_catalog_matches(item_description, catalog_df, threshold=0.6):
    """
    Find matches between item description and catalog entries
    """
    matches = []
    
    if pd.isna(item_description) or item_description == '':
        return matches
    
    # Clean the description
    description_clean = str(item_description).lower().strip()
    
    # Try exact matches first
    exact_matches = catalog_df[
        catalog_df['n4_nombre'].str.contains(description_clean, case=False, na=False) |
        catalog_df['nombre'].str.contains(description_clean, case=False, na=False)
    ]
    
    if len(exact_matches) > 0:
        return exact_matches
    
    # Try partial word matches
    words = description_clean.split()
    for word in words:
        if len(word) > 3:  # Only search for meaningful words
            word_matches = catalog_df[
                catalog_df['n4_nombre'].str.contains(word, case=False, na=False) |
                catalog_df['nombre'].str.contains(word, case=False, na=False)
            ]
            if len(word_matches) > 0:
                matches.extend(word_matches.to_dict('records'))
    
    return pd.DataFrame(matches).drop_duplicates() if matches else pd.DataFrame()

def expand_procurement_data(df):

    expanded_df = df.copy()

    expanded_df['lot_productCode'] = expanded_df['lot_productCode'].fillna('').astype(str)
    expanded_df[f'lot_productCode_cleaned'] = (expanded_df['lot_productCode'].str.strip().str.rstrip(',').str.split(','))

    expanded_df = expanded_df.explode(f'lot_productCode_cleaned')

    expanded_df['lot_productCode_cleaned'] = expanded_df['lot_productCode_cleaned'].str.strip()

    expanded_df = expanded_df[(expanded_df['lot_productCode_cleaned'] != '') & (expanded_df['lot_productCode_cleaned'].notna()) & (expanded_df['lot_productCode_cleaned'] != 'nan')]

    expanded_df['lot_productCode'] = expanded_df['lot_productCode_cleaned']
    expanded_df = expanded_df.drop(columns = [f'lot_productCode_cleaned'])

    expanded_df = expanded_df.reset_index(drop = True)
    expanded_df['is_exploded_row'] = True

    return expanded_df

def quantity_mapping_fixed(df, contract_quantities):
    """Map quantities using the original API URLs"""
    
    # Initialize quantity column
    if 'quantity' not in df.columns:
        df['quantity'] = None
        
    updated_rows = 0
    total_contracts_checked = 0
    
    # Group by contract URI for efficiency
    contract_groupings = df.groupby('tender_publications_lastcontract')
    
    for contract_uri, grouped_data_frame in tqdm(contract_groupings, desc='Mapping Contract Quantities'):
        total_contracts_checked += 1
        
        if contract_uri in contract_quantities:
            contract_items = contract_quantities[contract_uri]
            
            # Update each row in this contract group
            for index in grouped_data_frame.index:
                product_code = str(df.loc[index, 'lot_productCode'])
                
                if product_code in contract_items:
                    df.loc[index, 'quantity'] = contract_items[product_code]
                    updated_rows += 1
    
    print(f'Results:')
    print(f'- Total contract groups checked: {total_contracts_checked}')
    print(f'- Contracts with quantity data: {len([uri for uri in contract_groupings.groups.keys() if uri in contract_quantities])}')
    print(f'- Rows updated with quantities: {updated_rows}')
    
    return df