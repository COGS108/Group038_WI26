import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import seaborn as sns
import sys
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Optional

def parse_tags(tag_str):
    """Strip a stringified list into a Python list of tag strings."""
    if pd.isna(tag_str):
        return []
    return [t.strip().strip("'\"") for t in tag_str.strip("[]").split(",")]


def load_tags(years=range(2018, 2024)):
    """Load appid/tags/year from all yearly top-250 CSVs; parse tags into lists."""
    frames = pd.concat(
        [pd.read_csv(f"data/02-processed/{yr}_top250_final.csv")
             .assign(year=yr)[["appid", "tags", "year"]]
         for yr in years],
        ignore_index=True,
    )
    frames["tags"] = frames["tags"].apply(parse_tags)
    return frames


def explode_mode(df_tags, mode_tags=("singleplayer", "multiplayer", "co-op")):
    """Explode tag lists to one row per mode; keep only the three target modes.
    A single game can appear in multiple mode rows simultaneously."""
    exploded = df_tags.explode("tags").rename(columns={"tags": "mode"})
    return exploded[exploded["mode"].isin(mode_tags)].reset_index(drop=True)

"""
Section 3: Andy's implementation (for peak concurrent players)
- Appears to break modes into tuples, which was an idea we
  experimented with but decided otherwise.
"""
def explode_mode2(df_tags):
    mode_map = {        
        'singleplayer': 'singleplayer',
        'single-player': 'singleplayer',
        'multiplayer': 'multiplayer',
        'multi-player': 'multiplayer',
        'co-op': 'co-op',
        'coop': 'co-op',
        'cooperative': 'co-op'
    }

    exploded = df_tags.explode('tags').rename(columns={'tags': 'mode_raw'})
    exploded['mode_raw'] = exploded['mode_raw'].str.lower().str.strip()
    exploded['mode'] = exploded['mode_raw'].map(mode_map)
    exploded = exploded.dropna(subset=['mode'])

    combined = exploded.groupby(['appid', 'year'])['mode'].apply(
        lambda x: ', '.join(sorted(set(x)))
    ).reset_index()

    return combined

def explode_individual(df):
    df_exploded = df.copy()
    df_exploded['mode'] = df_exploded['mode'].str.split(', ')
    df_exploded = df_exploded.explode('mode')
    return df_exploded

def monthly_proportions(df):
    monthly = df.groupby(['month_dt', 'mode'], as_index=False)['peak_players'].sum()
    monthly['proportion'] = monthly['peak_players'] / monthly.groupby('month_dt')['peak_players'].transform('sum')
    return monthly

def period_proportion(df):
    df = df.copy()
    df['period'] = df['year'].apply(assign_period)
    
    cp = df.groupby(['period', 'mode'], as_index=False)['peak_players'].sum()

    cp['proportion'] = cp['peak_players'] / cp.groupby(['period'])['peak_players'].transform('sum')
    return cp

def assign_period(year):
    """Map a year to its COVID-era period label."""
    if 2018 <= year <= 2019:
        return "pre-COVID (2018-2019)"
    elif 2020 <= year <= 2021:
        return "COVID (2020-2021)"
    else:
        return "post-COVID (2022-2023)"
