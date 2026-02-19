import os
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd
import kagglehub
import shutil

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
    target_dir = "./data/00-raw"
    # https://www.kaggle.com/datasets/srgiomanhes/steam-games-dataset-2025
    # path = kagglehub.dataset_download("srgiomanhes/steam-games-dataset-2025")
    
    # https://www.kaggle.com/datasets/fronkongames/steam-games-dataset
    path = kagglehub.dataset_download("fronkongames/steam-games-dataset")

    os.makedirs(target_dir, exist_ok= True)

    # Ignore kagglehub download directory structure
    for file in os.listdir(path):
        shutil.move(os.path.join(path, file), target_dir)

    print("Steam 2025 dataset installed!")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rawdata_dir = os.path.join(script_dir, "..", "data", "00-raw")

    years = [2018, 2019, 2020, 2021, 2022, 2023]
    get_steam250(years, rawdata_dir)
    get_kaggle()

if __name__ == "__main__":
    main()