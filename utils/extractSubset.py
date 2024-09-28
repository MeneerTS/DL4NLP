import os
import shutil
import random
import json  # Import json module to parse JSON-formatted string
from pathlib import Path

# Define paths
home = str(Path.home())
subsets_path = os.path.join(home, "DL4NLP/subsets")  # Target folder for subsets
data_path = os.path.join(
    home, "DL4NLP/data"
)  # Source folder containing language folders
articles_path = os.path.join(
    home, "DL4NLP/articles.txt"
)  # Path to the articles.txt file

# Define paths for new folders within the 'subsets' folder
folders = {
    "en": os.path.join(subsets_path, "en_files"),
    "zh": os.path.join(subsets_path, "zh_files"),
    "de": os.path.join(subsets_path, "de_files"),
    "id": os.path.join(subsets_path, "id_files"),
    "ru": os.path.join(subsets_path, "ru_files"),
}

# Create new directories in the subsets folder
for folder in folders.values():
    os.makedirs(folder, exist_ok=True)

# Read and parse the list of articles from 'articles.txt'
with open(articles_path, "r") as file:
    content = file.read()
    # Parse the JSON-formatted string into a Python list
    articles = json.loads(content)

# Randomly select 50 articles
num_articles_to_select = min(50, len(articles))
selected_articles = random.sample(articles, num_articles_to_select)

# For each language folder, copy the selected article files
for lang in ["en", "zh", "de", "id", "ru"]:
    # Path to the current language folder in '../data'
    lang_folder_path = os.path.join(data_path, f"{lang}")

    # Check if the folder exists
    if not os.path.exists(lang_folder_path):
        print(f"Language folder {lang_folder_path} does not exist, skipping.")
        continue

    # Copy the selected articles to the destination folder
    for article in selected_articles:
        source_file = os.path.join(lang_folder_path, article)
        dest_file = os.path.join(folders[lang], article)
        if os.path.exists(source_file):
            shutil.copy(source_file, dest_file)
        else:
            print(f"Article {article} does not exist in language {lang}, skipping.")

print(
    "Randomly selected files have been copied to the subsets folder for each language."
)
