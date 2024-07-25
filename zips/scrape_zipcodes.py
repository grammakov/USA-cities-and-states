#!/usr/bin/env python

# ___________________________________________  #
# SCRAPE 5-DIGIT ZIP-CODES BASED ON CITY NAME  #
# By @hans-elliott99 (github)                  #
# ____________________________________________ #
#
# This process relies on an exhuastive list of cities from this GitHub repo: 
# https://github.com/grammakov/USA-cities-and-states/
# It uses City-State combinations to scrape zip-codes from: 
# https://codigo-postal.co/eeuu/
# The Zip-codes are at the city-level.
# For cities with multiple zips, all zips are included and separated by a space, like:
# |Holtsville|NY|...|00501 00544 11742|

import string
import os
import time
from math import ceil
# pip install:
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import multiprocessing as mp

# clean names
def _snake_case(name: str):
    return name.replace(' ', '_').lower()

def read_and_process_data():
    """Read in city data from GitHub, prep for use, and determine URLS"""
    # List of all cities in the US
    # https://github.com/grammakov/USA-cities-and-states/
    usa_cities = pd.read_csv(
        "https://github.com/grammakov/USA-cities-and-states/blob/master/us_cities_states_counties.csv?raw=true",
        delimiter="|"
        )
    # Clean names and isolate variables
    usa_cities.columns = [_snake_case(col) for col in usa_cities.columns]
    usa_cities['city_state'] = usa_cities['city']+"*"+usa_cities['state_full']

    allcities = usa_cities.loc[:, ['city', 'state_short', 'state_full', 'city_state']]
    # Drop duplicate rows
    allcities.drop_duplicates(inplace=True)
    allcities = allcities.reset_index(drop=True)
    # Determine url to scrape
    allcities['url'] = allcities.city_state.apply(_get_url_from_city)

    return allcities, usa_cities


def _get_url_from_city(city_state):
    """Use city and state name to determine the URL to search for zip-codes from."""
    url = "https://codigo-postal.co/eeuu/"
    # Prep city & state strings
    ##set of punctuation chars to remove
    rm_punct = [p for p in string.punctuation if p not in "-"] 
    loc = [city_state.split("*")[0], city_state.split("*")[1]]
    for i, obj in enumerate(loc):
        obj = str(obj).lower()
        obj = ''.join(c for c in obj if c not in rm_punct)
        obj = obj.replace(" ", "-")
        loc[i] = obj
    city, state = loc[0], loc[1]

    # Edge cases
    if "washington-dc" in state:
        state = "district-of-columbia"
    if "virgin-islands" in state:
        state = "united-states-virgin-islands"
    if "deptford" in city:
        city = "deptford-township"
    if "dewey-beach" in city:
        city = "dewey-bch"
    if "luke-air" in city:
        city = "luke-afb"
    
    url = url + state + "/" + city + "/"
    return url

def _get_zip_from_url(url):
    """Scrape the zip-codes belonging to a given city based on its url."""
    data = requests.get(url)
    html = BeautifulSoup(data.text, 'html.parser')
    table = html.select("table")[0].get_text().split(' ')
    zips = []
    for item in table:
        if len(item) == 5:
            try:
                int_item = int(item) #if we can convert to integer, then it's not a string
                zips.append(item)
            except:
                pass
    return ' '.join([z for z in zips])

def _get_zips(data):
    """Applies _get_zip_from_url across the dataframe."""
    df = data.copy()
    df['zips'] = df['url'].apply(_get_zip_from_url)
    return df

def _chunk_data(data, n_chunks):
    """Split data into chunks for parallel processing."""
    start_i = 0
    end_i = int(ceil(data.shape[0] / n_chunks))

    data_list = []
    for n in range(0, n_chunks):
        data_list.append(
            data[start_i : end_i].copy()
        )
        start_i = end_i
        end_i += int(ceil(data.shape[0] / n_chunks))
    
    return data_list

def mp_get_zips(data, n_workers):
    """Use multiprocessing to scrape zips in parallel."""
    data_list = _chunk_data(data, n_workers)

    with mp.Pool(n_workers) as p:
        results = p.map(_get_zips, data_list)
    return results

# Dictionary mapping city states to manually searched zip codes
missed_cities = {
    "Apo*US Armed Forces Europe" : "09014",
    "DPO*US Armed Forces Europe" : "09498",
    "FPO*US Armed Forces Europe" : "09596",
    "Fpo*US Armed Forces Europe" : "09596",
    "GPO*New York" : "10001",
    "NASA*Florida" : "32953",
    "Texhoma*Texas" : "73960",
    "Apo*US Armed Forces Pacific" : "96267",
    "DPO*US Armed Forces Pacific" : "96521",
    "FPO*US Armed Forces Pacific" : "96362",
    "Jb Phh*Hawaii" : "96818 96853 96860",
    "Palau*Palau" : "96940",
    "Pohnpei*Federated States of Micronesia": "96941",
    "Chuuk*Federated States of Micronesia": "96942",
    "Yap*Federated States of Micronesia": "96943",
    "Kosrae*Federated States of Micronesia": "96944",
    "Majuro*Marshall Islands" : "96960",
    "Ebeye*Marshall Islands" : "96970"    
}

if __name__=='__main__':
    start = time.time()
    N_WORKERS = mp.cpu_count() - 1 or 1

    print("Reading data from github...")
    allcities, cities_github = read_and_process_data()

    print(f"Scraping ZIP codes with {N_WORKERS} workers...")
    results = mp_get_zips(allcities, N_WORKERS)

    # Concatenate results
    cities = pd.concat(results)

    # Impute missed cases
    ## First convert blank entries to NA
    cities.zips = cities.zips.replace(r'^\s*$', np.nan, regex=True)
    ## Then impute with the found zip-codes (See above)
    cities.zips = cities.zips.fillna(cities.city_state.map(missed_cities))

    # Keep only desired columns
    city2zip = cities[['city', 'state_short', 'state_full', 'zips']]

    # Save city to zipcode mapping to csv
    city2zip.to_csv(
        os.path.abspath("./city2zip.csv"),
        index=False
    )
    print(f"Saved to CSV. Runtime: {time.time() - start :.3f}s")

    # -----
    # Convert format for consistency with GitHub data
    ## Merge scraped zip codes onto the data pulled from github
    cities = cities[['city_state', 'zips']]
    merged = cities.merge(cities_github)
    ## Keep desired columns and rename for consistency
    merged = merged[['city', 'state_short', 'state_full', 'county', 'city_alias', 'zips']]
    merged.columns = ['City', 'State short', 'State full', 'County', 'City alias', 'City zip codes']
    ## Save
    merged.to_csv(
        "us_cities_states_counties_zips.csv",
        sep="|",
        index=False
    )
