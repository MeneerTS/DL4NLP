import os, tarfile, shutil, random
from pathlib import Path

# Define paths
home = str(Path.home())
desktop_path = os.path.join(home, "Desktop")
download_path = os.path.join(home, "Downloads")
# Paths for new folders on the desktop
folders = {
    "en": os.path.join(desktop_path, "en_files"),
    "nl": os.path.join(desktop_path, "nl_files"),
    "it": os.path.join(desktop_path, "it_files"),
}

# Create new directories on the desktop
for folder in folders.values():
    os.makedirs(folder, exist_ok=True)

documents_path = os.path.join(download_path, "documents.tgz")
# Open the tar archive
with tarfile.open(documents_path, "r:gz") as tar:
    # Initialize dictionary to store file names for each language
    file_names_by_lang = {}
    for lang in ["en", "nl", "it"]:
        folder_path_in_tar = f"split/{lang}/"
        # Get all .txt files in the language folder
        file_names = [
            name
            for name in tar.getnames()
            if name.startswith(folder_path_in_tar) and name.endswith(".txt")
        ]
        # Extract the base file names (excluding the 'split/lang/' prefix)
        base_file_names = [os.path.basename(name) for name in file_names]
        file_names_by_lang[lang] = set(base_file_names)
    # Find the common file names across all languages
    common_file_names = set.intersection(*file_names_by_lang.values())
    # Randomly select 50 common file names
    selected_file_names = random.sample(
        common_file_names, min(len(common_file_names), 50)
    )
    # For each selected file name, extract and move the files
    for file_name in selected_file_names:
        for lang in ["en", "nl", "it"]:
            file_in_tar = f"split/{lang}/{file_name}"
            tar.extract(file_in_tar, path="tmp")  # Extract to a temporary directory
            source_file = os.path.join("tmp", file_in_tar)
            dest_file = os.path.join(folders[lang], file_name)
            shutil.move(source_file, dest_file)

# Clean up the temporary directory
shutil.rmtree("tmp")
print("50 common files from 'en', 'nl', and 'it' have been moved to your desktop.")
