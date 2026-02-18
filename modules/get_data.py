import os
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd

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
    for year in tqdm(years):
        url = f'https://steam250.com/{year}'

        try:
            html = requests.get(url).text
            soup = BeautifulSoup(html, 'html.parser')

            section = soup.select_one("section.applist.compact")
            rows = section.find_all("div", id=True) 

            records = []

            for row in rows:
                try:
                    rank_div = row.find_all('div', recursive=False)[0]
                    texts = [t.strip() for t in rank_div.contents if isinstance(t, str) and t.strip()]
                    if texts:
                        rank = int(texts[0])
                    else:
                        continue
                    
                    name_tag = row.select_one("a[title]")
                    game = name_tag.text.strip()

                    company_tag = row.select_one("span.company")
                    if company_tag:
                        company = company_tag.text.strip()
                    else:
                        company = ""

                    score_tag = row.select_one("span.score")
                    if score_tag:
                        score = score_tag.text.strip()
                    else:
                        score = ""

                    rating_tag = row.select_one("span.rating")
                    vote_tag = row.select_one("span.votes")

                    if rating_tag:
                        rating = rating_tag.text.strip()
                        if vote_tag:
                            rating = f"{rating} {vote_tag.text.strip()}"
                    records.append({'Rank': rank, 'Game': game, 'Company': company, 'Score': score, 'Rating': rating})
                except:
                    continue
            
            df = pd.DataFrame(records)
            output = os.path.join(destination_directory, f'{year}_top250.csv')
            df.to_csv(output, index=False)
            print(f" saved file {year}_top250.csv")
        
        except Exception as e:
            print(f" error find on {year}: {e}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rawdata_dir = os.path.join(script_dir, "..", "data", "00-raw")

    years = [2018, 2019, 2020, 2021, 2022, 2023]
    get_steam250(years, rawdata_dir)


