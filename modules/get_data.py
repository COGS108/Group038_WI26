import os
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd
import kagglehub
import shutil
import time
import hashlib
from typing import Optional
from pathlib import Path


def get_raw(file_list, destination_directory):
    """
    Downloads a list of files to a specified destination directory with progress bars.

    Args:
        file_list (list): A list of dictionaries, where each dictionary
                          contains 'url' (the URL of the file) and 'filename'
                          (the desired local filename).
        destination_directory (str): The path to the directory where files
                                     should be saved.
    """
    if not os.path.exists(destination_directory):
        print(f"Error directory {destination_directory} does not exist")
        return

    for file_info in tqdm(file_list, desc="Overall Download Progress"):
        url = file_info['url']
        filename = file_info['filename']
        local_filepath = os.path.join(destination_directory, filename)

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB

            with open(local_filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True,
                          desc=f"Downloading {filename}", leave=False) as pbar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            pbar.update(len(chunk))
            print(f"Successfully downloaded: {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for {filename}: {e}")

def get_steam250(years, destination_directory):
    for year in years:
        url = f'https://steam250.com/{year}'
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')

        section = soup.select_one("section.applist.compact")
        rows = section.find_all("div", id=True)

        records = []

        for row in rows:
            try:
                rank_div = row.find_all("div", recursive=False)[0]

                # Ignore rank change stats e.g. "+2" or "-1"
                texts = [t.strip() for t in rank_div.contents if isinstance(t, str) and t.strip()]
                if texts:
                    rank = int(texts[0])
                else:
                    continue

                name_tag = row.select_one("a[title]")
                name = name_tag.text.strip()

                app_link = name_tag["href"]
                appid = int(app_link.split("/")[-1])

                rating = row.select_one("span.rating").text.strip().replace("%","")
                rating = float(rating)

                votes_raw = row.select_one("span.votes")["title"]
                votes = votes_raw.replace(",", "")
                votes = votes.replace("reviews", "")
                votes = votes.replace("\"", "")
                votes = int(votes)

                records.append({
                    "rank": rank,
                    "appid": appid,
                    "name": name,
                    "rating": rating,
                    "num_votes": votes
                })

            except Exception as e:
                print(e)
                continue

        curr_df = pd.DataFrame(records)
        curr_file_name = os.path.join(destination_directory, f'{year}_top250.csv') 

        curr_df.to_csv(curr_file_name, index=False)

    # Done message
    print("Top 250 Steam games for 2018-2023 installed!")

def get_kaggle(): 
    target_dir = "./data/00-raw/"
    # https://www.kaggle.com/datasets/srgiomanhes/steam-games-dataset-2025
    # path = kagglehub.dataset_download("srgiomanhes/steam-games-dataset-2025")
    
    # Source: https://www.kaggle.com/datasets/fronkongames/steam-games-dataset
    path = kagglehub.dataset_download("fronkongames/steam-games-dataset")

    os.makedirs(target_dir, exist_ok= True)

    # Ignore kagglehub download directory structure
    for file in os.listdir(path):
        shutil.move(os.path.join(path, file), target_dir)

    print("Steam dataset installed!")



"""
DORJE'S CODE STARTS HERE
"""
#### Utility Functions Overview
# The helper functions below standardize small repeated tasks so the collection pipeline is easier to read and debug.  
# - `clean_num` converts numeric text scraped from HTML (including commas and dash placeholders) into floats.  
# - `build_game_url` creates a SteamCharts URL from a game `appid`.  
# - `cache_key_for_url` creates a deterministic filename-safe cache key from a URL so cached HTML pages can be reused across runs.
# SteamCharts collection helpers + pipeline 

# Base URL pattern for SteamCharts game pages
# inject app_id into {app_id}, e.g. app_id=730 -> https://steamcharts.com/app/730
BASE_URL = "https://steamcharts.com/app/{appid}"

# Apprently some sites block requests that do not provide a browser-like user agent.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SteamChartsYearScraper/1.0)"
}

# Some utlity helpers! ==========================================================================

def clean_num(value: str) -> Optional[float]:
    """
    Convert numeric text to float.

    Handles:
    - commas: "12,345.6" -> 12345.6
    - blanks/dashes -> None
    - invalid values -> None
    """
    if value is None:
        return None

    text = str(value).strip().replace(",", "")
    if text in {"", "-", "—"}:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def build_game_url(appid: int) -> str:
    """
    Build SteamCharts URL for one appid.
    Example: appid=730 -> "https://steamcharts.com/app/730"
    """
    return BASE_URL.format(appid=int(appid))



def cache_key_for_url(url: str) -> str:
    """
    Build deterministic cache filename key from URL.
    Using md5 keeps filenames short and filesystem-safe.
    Why this exists:
    - URL text may not be ideal as a filename.
    - Hash gives stable and filesystem-safe names.
    """
    return hashlib.md5(url.encode("utf-8")).hexdigest()
#### Input Loading and Validation
# Before scraping SteamCharts, we validate each yearly input CSV to ensure it has the expected schema: `rank`, `name`, and `appid`.  
# This step enforces consistent data types, removes empty names, drops duplicates, and sorts by rank so processing is deterministic.  
# Failing early on malformed input helps avoid harder-to-diagnose errors later in the scraping pipeline.
# Input and loading validation ==================================================================
def load_input_csv(file_path: Path) -> pd.DataFrame:
    """
    Load and validate one input CSV.
    (Probably not neccessary, but I think it's good practice just in case)

    Required columns:
    - rank
    - name
    - appid

    Returns:
    - Cleaned DataFrame with normalized dtypes:
      rank:int, name:str, appid:int
    """

    # Read CSV (consider encoding="utf-8-sig")
    df = pd.read_csv(file_path, encoding="utf-8-sig")

    # Validate required columns
    required = {"rank", "name", "appid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{file_path.name} is missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    # Keep only needed columns in predictable order 
    df = df[["rank", "name", "appid"]].copy()

    # Convert numeric fields and fail LOUDLY if invalid
    df["rank"] = pd.to_numeric(df["rank"], errors="raise").astype(int)
    df["appid"] = pd.to_numeric(df["appid"], errors="raise").astype(int)

    # 4) Strip whitespace on name
    df["name"] = df["name"].astype(str).str.strip()

    # Remove rows with empty names 
    df = df[df["name"] != ""].copy()

    # drop duplicates on rank+appid
    df = df.drop_duplicates(subset=["rank", "appid"]).reset_index(drop=True)

    # Return cleaned DataFrame
    # Sort by rank for deterministic processing
    df = df.sort_values("rank").reset_index(drop=True)


    return df
#### Network Requests and HTML Caching
# The scraper first checks whether a game's HTML page is already cached locally. If so, it reads from cache; otherwise, it fetches the page from SteamCharts and stores a local copy.  
# This reduces repeated web requests, improves reproducibility across reruns, and speeds up development/testing.  
# A short delay is added between live requests to avoid overloading the source website.
# Network + cache =================================================================================
def get_game_page_html(
    appid: int,
    session: requests.Session,
    cache_dir: Path,
    use_cache: bool = True,
    request_delay_sec: float = 0.6,
) -> str:
    """
    Return HTML for one game page, using cache when available.

    Flow:
    1) Build game URL from appid
    2) Compute cache filename from URL hash
    3) If cache exists and use_cache=True -> return cached HTML
    4) Else fetch from network, save cache, sleep briefly, return HTML
    """

    # Ensure cache_dir exists
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Compute URL and cache filename
    url = build_game_url(appid)
    cache_file = cache_dir / f"{cache_key_for_url(url)}.html"

    # If cache hit and use_cache: read + return
    if cache_file.exists() and use_cache:
        return cache_file.read_text(encoding="utf-8")

    # Live request path
    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    html = resp.text

    # Save to cache for future runs
    cache_file.write_text(html, encoding="utf-8")

    # Don't attack our lord and savior GabeN with rapid-fire requests
    time.sleep(request_delay_sec)

    return html


# HTML parsing ===================================================================================


def parse_year_data_from_html(html: str, target_year: int) -> list[dict]:
    """
    Parse SteamCharts monthly table from one app page, filtered to target year.

    Input:
    - html: raw page HTML
    - target_year: year to keep (e.g., 2021)

    Output row shape:
    {
      "month": "YYYY-MM",
      "avg_players": float|None,
      "peak_players": float|None
    }

    Why this exists:
    - Pure parser function (HTML in -> structured rows out).
    - Easy to test independently from I/O.
    """

    # Parse html with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    parsed_rows = []

    # Select monthly table rows (table.common-table tbody tr)
    rows = soup.select("table.common-table tbody tr")

    # For each row, parse month/avg/peak
    for tr in rows: 
        tds = tr.find_all("td")

        # Monthly rows should have at least 5 columns
        # Month | Avg. Players | Gain | Gain % | Peak Players
        if len(tds) < 5:
            continue

        month_text = tds[0].get_text(" ", strip=True)

        # Skip "Last 30 Days"
        if month_text.lower() == "last 30 days":
            continue

        # Parse month text with pd.to_datetime(..., format="%B %Y")
        month_dt = pd.to_datetime(month_text, format="%B %Y", errors="coerce")
        if pd.isna(month_dt):
            continue

        # Keep rows where parsed year == target_year
        if int(month_dt.year) != int(target_year):
            continue

        # Clean numeric fields with clean_num()
        avg_text = tds[1].get_text(" ", strip=True)
        peak_text = tds[4].get_text(" ", strip=True)

        # Return rows sorted by month asc
        parsed_rows.append({
            "month": month_dt.strftime("%Y-%m"),
            "avg_players": clean_num(avg_text),
            "peak_players": clean_num(peak_text),
        })

    # Keep output in chronological order
    parsed_rows.sort(key=lambda r: r["month"])
    return parsed_rows


#### Yearly Collection Logic
# For each year, we read the corresponding top-250 appid file, fetch each game's SteamCharts page, and parse only rows that match the target year.  
# Each output row is assigned a status:
# - `ok` if monthly rows were parsed successfully,
# - `no_data_for_year` if the page loaded but no rows matched the year,
# - `request_error` for HTTP/network failures,
# - `parse_error` for HTML parsing issues.
# This status field helps us audit data quality before analysis.
# year collection ====================================================================================
def collect_one_year(
    input_csv: Path,
    year: int,
    cache_dir: Path,
    use_cache: bool = True,
    request_delay_sec: float = 0.6,
) -> pd.DataFrame:
    """
    Collect SteamCharts data for one year's input list.

    Steps:
    - Load CSV (rank, name, appid)
    - For each game:
      - Fetch or read cached HTML
      - Parse target-year monthly rows
      - Emit result rows with status labels

    Status values:
    - "ok"               : parsed monthly rows exist
    - "no_data_for_year" : page loaded, but no rows for that year
    - "request_error"    : failed HTTP request
    - "parse_error"      : page fetched but parse failed
    """

    # games_df = load_input_csv(input_csv)
    games_df = load_input_csv(input_csv)

    # Initialize out_rows = []
    out_rows = []
    total = len(games_df)

    # Create requests.Session()
    with requests.Session() as session:
        for idx, row in games_df.iterrows():
            rank = int(row["rank"])
            name = row["name"]
            appid = int(row["appid"])

            # Fetch HTML (cache first)
            try: 
                html = get_game_page_html(
                    appid=appid,
                    session=session,
                    cache_dir=cache_dir,
                    use_cache=use_cache,
                    request_delay_sec=request_delay_sec,
                )
            except Exception:
                out_rows.append({
                    "year": year,
                    "rank": rank,
                    "name": name,
                    "appid": appid,
                    "month": None,
                    "avg_players": None,
                    "peak_players": None,
                    "status": "request_error",
                })
                print(f"[{idx+1}/{total}] {name} (appid={appid}): request error")
                continue

            # Parse only rows for target year
            try:
                parsed_rows = parse_year_data_from_html(html, target_year=year)
            except Exception:
                out_rows.append({
                    "year": year,
                    "rank": rank,
                    "name": name,
                    "appid": appid,
                    "month": None,
                    "avg_players": None,
                    "peak_players": None,
                    "status": "parse_error",
                })
                print(f"[{idx+1}/{total}] {name} (appid={appid}): parse error")
                continue

            # no rows found for this yeaer 
            if not parsed_rows:
                out_rows.append({
                    "year": year,
                    "rank": rank,
                    "name": name,
                    "appid": appid,
                    "month": None,
                    "avg_players": None,
                    "peak_players": None,
                    "status": "no_data_for_year",
                })
                print(f"[{idx+1}/{total}] {name} (appid={appid}): no data for year")
                continue

            # Found rows: attach metadata 
            for pr in parsed_rows:
                out_rows.append({
                    "year": year,
                    "rank": rank,
                    "name": name,
                    "appid": appid,
                    "month": pr["month"],
                    "avg_players": pr["avg_players"],
                    "peak_players": pr["peak_players"],
                    "status": "ok",
                })

            print(f"[{idx+1}/{total}] {name} ({appid}) -> ok ({len(parsed_rows)} months)")

    result_df = pd.DataFrame(
        out_rows,
        columns=[
            "year", 
            "rank", 
            "name", 
            "appid",
            "month", 
            "avg_players", 
            "peak_players",
            "status",
        ],)

    # Sort for readability (rank, then month)
    result_df = result_df.sort_values(
                    by=["rank", "month"], 
                    na_position="last"
                ).reset_index(drop=True)

    return result_df


def collect_year_range(
    start_year: int,
    end_year: int,
    input_dir: Path,
    input_pattern: str,      # e.g. "{year}_top250_ids.csv"
    output_dir: Path,
    cache_dir: Path,
    use_cache: bool = True,
    request_delay_sec: float = 0.6,
    write_combined: bool = True,
) -> pd.DataFrame:
    """
    Run collection across a year range using predictable filenames.

    For each year:
    - Build input file path from input_pattern
    - Skip year if file missing
    - Collect year data
    - Write per-year CSV

    Optionally:
    - Combine all years into one DataFrame + CSV
    """

    # Ensure output_dir exists
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_parts = []

    for year in range(start_year, end_year + 1):
    #   build input_csv path from pattern
        input_csv = input_dir / input_pattern.format(year=year)

    #   if missing file: print skip and continue
        if not input_csv.exists():
            print(f"{year} SKIP - missing input file:{input_csv}")
            continue

        year_df = collect_one_year(
            input_csv=input_csv,
            year=year,
            cache_dir=cache_dir,
            use_cache=use_cache,
            request_delay_sec=request_delay_sec,
        )
    #   write year_df to output_dir / f"steamcharts_{year}_top250.csv"
        year_out_path = output_dir / f"steamcharts_{year}_top250.csv"
        year_df.to_csv(year_out_path, index=False)
        print(f"[{year}] wrote {year_out_path} ({len(year_df)} rows)")

        status_counts = year_df["status"].value_counts(dropna=False)
        print(f"[{year}] status summary:\n{status_counts.to_string()}")

    #   append year_df to all_parts
        all_parts.append(year_df)

    # If nothing processed, return empty with expected schema
    if not all_parts:
        return pd.DataFrame(columns=[
            "year", "rank", "name", "appid", "month",
            "avg_players", "peak_players", "status"
        ])

    combined_df = pd.concat(all_parts, ignore_index=True)

    if write_combined:
        combined_out_path = output_dir / f"steamcharts_{start_year}_{end_year}_combined.csv"
        combined_df.to_csv(combined_out_path, index=False)
        print(f"\nWrote combined file: {combined_out_path} ({len(combined_df)} rows)")

    return combined_df

#### Status filtering
# Not all of the requests are expected to go through perfectly, so statuses are attatched to them so we can easily pick out which ones are missing data. 

# This function takes a year, loads that year’s interim SteamCharts CSV, filters to rows where `status == "ok"`, and writes the filtered result to `data/02-processed`.

from pathlib import Path
import pandas as pd

def keep_only_ok_status(
    year: int,
    interim_dir: Path = Path("./data/01-interim"),
    processed_dir: Path = Path("./data/02-processed"),
) -> Path:
    """
    Reads one yearly interim SteamCharts CSV and writes an ok-only version.

    Input:
      data/01-interim/steamcharts_{year}_top250.csv
    Output:
      data/02-processed/steamcharts_{year}_top250_ok.csv

    Returns the output path.
    """
    input_path = interim_dir / f"steamcharts_{year}_top250.csv"
    output_path = processed_dir / f"steamcharts_{year}_top250_ok.csv"

    df = pd.read_csv(input_path)

    if "status" not in df.columns:
        raise ValueError(f"'status' column not found in {input_path}")

    df_ok = df[df["status"] == "ok"].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_ok.to_csv(output_path, index=False)

    return output_path



def make_combined_ok_file(
    start_year: int,
    end_year: int,
    processed_dir: Path = Path("./data/02-processed"),
) -> Path:
    """
    Combines the yearly ok-only files into one ok-only combined file.

    Inputs:
      data/02-processed/steamcharts_{year}_top250_ok.csv
    Output:
      data/02-processed/steamcharts_{start}_{end}_ok.csv

    Returns the output path.
    """
    parts = []

    for year in range(start_year, end_year + 1):
        fp = processed_dir / f"steamcharts_{year}_top250_ok.csv"
        if fp.exists():
            parts.append(pd.read_csv(fp))

    if not parts:
        raise FileNotFoundError("No yearly ok-only files found to combine.")

    combined_ok = pd.concat(parts, ignore_index=True)

    out_path = processed_dir / f"steamcharts_{start_year}_{end_year}_ok.csv"
    combined_ok.to_csv(out_path, index=False)

    return out_path

#### Quick Data Quality and Summary
# This section loads the combined SteamCharts output and reports high-level checks used before modeling:
# - available columns,
# - dataset size,
# - status counts from collection,
# - period-level descriptive summaries for `avg_players` and `peak_players` using only `status == "ok"` rows.
# These checks provide a compact overview of data completeness and trend direction across pre-COVID, COVID, and post-COVID periods.
def report_steamcharts_summary(csv_path: Path, label: str = "") -> None:
    """
    Print a compact data-quality + descriptive summary for a SteamCharts scrape CSV.

    Reports:
    - Columns
    - Overall size
    - Status counts (if column exists)
    - Period-level summary of avg_players / peak_players
      (uses status=='ok' rows when status column exists)
    """
    df = pd.read_csv(csv_path)

    header = f"=== {label} ===" if label else "==="
    print(f"\n{header}")
    print(f"File: {csv_path}")
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    print("\nColumns:")
    print(df.columns.tolist())

    # Status counts (only if present)
    if "status" in df.columns:
        print("\nStatus counts:")
        print(df["status"].value_counts(dropna=False).to_string())

    # Period-level summary (descriptive only)
    required = {"year", "avg_players", "peak_players"}
    if required.issubset(df.columns):
        tmp = df.copy()

        # light coercion for summary calculations only
        tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce")
        tmp["avg_players"] = pd.to_numeric(tmp["avg_players"], errors="coerce")
        tmp["peak_players"] = pd.to_numeric(tmp["peak_players"], errors="coerce")

        # If status exists, restrict to ok for summaries
        if "status" in tmp.columns:
            tmp = tmp[tmp["status"] == "ok"].copy()

        def period_label(y):
            if pd.isna(y):
                return "unknown"
            y = int(y)
            if 2018 <= y <= 2019:
                return "pre_covid"
            if 2020 <= y <= 2021:
                return "covid"
            if 2022 <= y <= 2023:
                return "post_covid"
            return "other"

        tmp["period"] = tmp["year"].apply(period_label)

        period_summary = (
            tmp.groupby("period", as_index=False)
               .agg(
                   n_rows=("period", "size"),
                   avg_players_mean=("avg_players", "mean"),
                   avg_players_median=("avg_players", "median"),
                   peak_players_mean=("peak_players", "mean"),
                   peak_players_median=("peak_players", "median"),
               )
               .sort_values("period")
        )

        print("\nPeriod-level summary (avg/peak players):")
        print(period_summary.to_string(index=False))

#### Main Run Configuration
# `main()` centralizes the user-editable run settings (year range, input pattern, output directory, cache directory, and delay settings).  
# This keeps the rest of the pipeline stable while making it easy to rerun collection with different folders or year ranges.  
# The run writes one CSV per year and a combined CSV for downstream summary and analysis, then prints a summary of the shape of the data, tidys the data, then prints the summary once more.
def steamcharts_main():
    start_year = 2018
    end_year = 2023

    input_dir = Path("./data/02-processed")
    input_pattern = "{year}_top250_final.csv"
    output_dir = Path("./data/01-interim")
    cache_dir = Path("./data/00-raw/steamcharts_cache")

    # Run scraper to create interim per-year + combined
    collect_year_range(
        start_year=start_year,
        end_year=end_year,
        input_dir=input_dir,
        input_pattern=input_pattern,
        output_dir=output_dir,
        cache_dir=cache_dir,
        use_cache=True,
        request_delay_sec=0.6,
        write_combined=True,
    )

    # Print BEFORE summary (combined interim)
    combined_before = output_dir / f"steamcharts_{start_year}_{end_year}_combined.csv"
    report_steamcharts_summary(combined_before, label="Combined BEFORE status filtering")

    # Create yearly ok-only files
    for year in range(start_year, end_year + 1):
        keep_only_ok_status(year, interim_dir=output_dir, processed_dir=Path("./data/02-processed"))

    # Create combined ok-only file
    combined_after = make_combined_ok_file(start_year, end_year, processed_dir=Path("./data/02-processed"))

    # Print AFTER summary (combined ok-only)
    report_steamcharts_summary(combined_after, label="Combined AFTER status filtering (ok-only)")
"""
DORJE'S CODE STARTS HERE
"""

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rawdata_dir = os.path.join(script_dir, "..", "data", "00-raw")

    years = [2018, 2019, 2020, 2021, 2022, 2023]
    get_steam250(years, rawdata_dir)
    get_kaggle()
    steamcharts_main()

if __name__ == "__main__":
    main()