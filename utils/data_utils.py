import os, wget, tarfile
from tqdm import tqdm
from pathlib import Path
from utils.constants import *
from collections import defaultdict
from lingua import LanguageDetectorBuilder


def download_nmt(link: str = DATA_LINK):
    """
    Downloads a dataset (in this case, News Commentary v16).

    Arguments:
    link (str): Link to the dataset. Should be a link that when clicked, immediately
    commences with the download.
    """

    # Check the file presence in the data directory
    if os.listdir(DATA_PATH) == [".gitkeep"]:
        print("Downloading data...")
        wget.download(link, out=DATA_PATH, bar=wget.bar_adaptive)
        print("\nFinished downloading!\n")

        # Unzipping the files
        print("Extracting data...")
        with tarfile.open(TAR_PATH) as tar:
            members = tar.getmembers()
            for member in tqdm(members, total=len(members), desc="Extracting"):
                tar.extract(member, path=DATA_PATH)

        print("Finished extracting!\n")

        # Moving everything up a folder (out of "split")
        print("Reorganizing the directories...")
        files_to_move = [
            (
                Path(path) / file,
                Path(DATA_PATH) / Path(path).relative_to(SPLIT_PATH) / file,
            )
            for path, _, files in os.walk(SPLIT_PATH)
            if get_dir_depth(path) >= 3
            for file in files
        ]

        for src, dst in tqdm(files_to_move, desc="Moving files"):
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)

        print("Finished reorganizing!\n")

        # Cleaning up
        os.remove(SPLIT_PATH)
        os.remove(TAR_PATH)
        print("\nProcess finished!")

    else:
        print("Dataset already downloaded.")


def get_dir_depth(path: str):
    """
    Counts how many layers a directory contains.
    For example: "data//ar//text.txt" is 3 layers deep.

    Arguments:
    path (str): Directory to check the depth of.

    Returns:
    The depth of the directory.
    """

    # Normalize the path to handle different path separators
    normalized_path = os.path.normpath(path)
    path_components = normalized_path.split(os.sep)

    # Remove empty components
    path_components = [component for component in path_components if component]

    return len(path_components)


def get_article_text(directory: str, remove_p: bool = False):
    """
    Retrieves and cleans text from an article.

    Arguments:
    directory (str): The .txt article file.
    remove_p (bool): Whether to remove the <P> breaks in the text.

    Returns:
    The contents of that article file, cleaned.
    """
    # Get the text contents
    with open(directory, "r") as f:
        text = f.read()

    # Clean text
    text = text.replace("<HEADLINE>\n", "")
    text = text.replace("\n", " ")

    # Removing <P>
    if remove_p:
        text = text.replace(" <P> ", "\n")

    return text


def clean_data(directory: str = DATA_PATH):
    """
    Checks if all paragraphs in all folders are in the same language.
    Necessary because the News Commentary dataset has some faulty entries
    (i.e., many files have some English paragraphs).

    For now, these files will be removed (NOT IMPLEMENTED YET).

    Arguments:
    directory (str): Folder containing all the data.
    """
    # Setting up the language detector
    detector = (
        LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()
    )

    problem_counts = defaultdict(int)

    for dirpath, _, filenames in os.walk(directory):

        if get_dir_depth(dirpath) > 1:
            # Get the language from the directory name
            lang_code = dirpath[-2:]
            lang_target = LANG_DICT[lang_code]
            print(f"On folder: {lang_code}")

            for filename in filenames:
                if filename.endswith(".txt"):

                    # Get the language from the directory name
                    fullpath = os.path.join(dirpath, filename)
                    text = get_article_text(fullpath)
                    sentences = text.split("<P>")[1:]  # Ignore titles for now

                    # print(fullpath)

                    for sentence in sentences:

                        if "<SOURCE" in sentence or sentence == " ":
                            continue

                        lang_curr = detector.detect_language_of(sentence)
                        # print(sentence, "|", lang_curr)
                        if lang_target != lang_curr:
                            print(f"File contains more than 1 language. Removing...")
                            problem_counts[lang_code] += 1
                            break

                    # print()

    print(problem_counts)

    return
