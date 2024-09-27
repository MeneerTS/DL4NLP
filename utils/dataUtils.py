import os, wget, tarfile, jieba, fugashi
from re import sub, findall
from tqdm import tqdm
from pathlib import Path
from utils.constants import *
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


def get_article_text(
    directory: str,
    remove_h: bool = True,
    remove_p: bool = False,
    remove_n: bool = True,
):
    """
    Retrieves and cleans text from an article.

    Arguments:
    directory (str): The .txt article file.
    remove_p (bool): Whether to remove the <P> breaks in the text.
    remove_n (bool): Whether to remove line breaks (\n) from the text.

    Returns:
    The contents of that article file, cleaned.
    """
    # Get the text contents
    with open(directory, "r") as f:
        text = f.read()

    # Clean text
    if remove_h:
        text = text.replace("<HEADLINE>\n", "")

    # Removing \n
    if remove_n:
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
    if os.path.isfile(DONE_DIR):
        print("Files already cleaned.")
        return

    detector = LanguageDetectorBuilder.from_all_languages().build()
    author_filter = r'<SOURCE\s+TRANSLATOR="[^"]+">'

    # Count total files for overall progress
    total_files = sum(
        len([f for f in files if f.endswith(".txt")])
        for root, _, files in os.walk(directory)
        if get_dir_depth(root) > 1
    )

    with tqdm(total=total_files, desc="Overall Progress") as pbar:

        for dirpath, _, filenames in os.walk(directory):

            if get_dir_depth(dirpath) > 1:
                lang_code = os.path.basename(dirpath)
                lang_target = LANG_DICT.get(lang_code)

                if lang_target is None:
                    print(f"Warning: Unknown language code {lang_code}")
                    continue

                for filename in filenames:
                    if filename.endswith(".txt"):
                        fullpath = os.path.join(dirpath, filename)
                        try:
                            text = get_article_text(fullpath)

                        except Exception as e:
                            print(f"Error reading file {fullpath}: {e}")
                            continue

                        sentences = text.split(" <P> ")
                        clean_sentences = []

                        for sentence in sentences:

                            # Extra cleanup for insurance
                            sentence = sub("<P>", "", sentence)

                            if "<SOURCE" in sentence:
                                sentence = sub(author_filter, "", sentence)
                            lang_curr = detector.detect_language_of(sentence)

                            if (
                                lang_target != lang_curr
                                and lang_curr not in LANG_REL.get(lang_code, [])
                                and lang_curr is not None
                                and len(sentence) > 20
                                and lang_target is not Language.ENGLISH
                            ):
                                continue

                            clean_sentences.append(sentence)

                        if clean_sentences:
                            title = clean_sentences[0]
                            text_content = "\n".join(clean_sentences[1:])
                            full_text = f"{title}\n\n{text_content}"

                            try:
                                with open(fullpath, "w") as f:
                                    f.write(full_text)

                            except Exception as e:
                                print(f"Error writing to file {fullpath}: {e}")

                        pbar.update(1)

    with open(DONE_DIR, "w") as f:
        f.write("done")


# Extra utils for getting dataset statistics
def count_tokens_in_document(text, lang_code, use_period: bool = True):
    """
    Counts the number of tokens in the given text, handling tokenization
    for different languages appropriately.
    Arguments:
    text (str): The text to tokenize.
    lang_code (str): The language code of the text.
    use_period (bool): Whether to count periods as words.
    Returns:
    int: The number of tokens.
    """
    # For Chinese
    if lang_code == "zh":
        tokens = jieba.lcut(text)
    # For Japanese
    elif lang_code == "ja":
        tagger = fugashi.Tagger()
        tokens = [word.surface for word in tagger(text)]
    # For other languages (split on whitespace and periods)
    else:
        # Split on whitespace and periods, keeping the periods
        tokens = findall(r"\S+|\s+|\.", text)
        # Remove empty strings from the result
        tokens = [token for token in tokens if token.strip()]

    # Period removal
    if not use_period:
        tokens = [token for token in tokens if token not in {".", "。", "．", "｡"}]

    return len(tokens)


def count_document_lengths(directory: str = DATA_PATH):
    """
    Counts the lengths of all documents in terms of tokens in the folder,
    handling different tokenization needs for Chinese and Japanese.

    Arguments:
    directory (str): Folder containing all the data.
    """
    # To hold the counts
    total = 0

    # Count total files for overall progress
    total_files = sum(
        len([f for f in files if f.endswith(".txt")])
        for root, _, files in os.walk(directory)
        if get_dir_depth(root) > 1
    )

    with tqdm(total=total_files, desc="Overall Progress") as pbar:

        for dirpath, _, filenames in os.walk(directory):

            if get_dir_depth(dirpath) > 1:
                lang_code = os.path.basename(dirpath)

                for filename in filenames:
                    if filename.endswith(".txt"):
                        fullpath = os.path.join(dirpath, filename)
                        try:
                            text = get_article_text(fullpath)
                        except Exception as e:
                            print(f"Error reading file {fullpath}: {e}")
                            continue

                        # Count the number of tokens in the document
                        total += count_tokens_in_document(text, lang_code)

                        pbar.update(1)

    return total


# For dataset generation
def extract_title_and_sentence(text):
    # Extract title: First element in list
    sents = text.split("\n")
    sents = list(filter(None, sents))
    title, sentence = sents[0], sents[1]

    return title, sentence
