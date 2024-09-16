import os
from torch.utils.data import Dataset
from utils.constants import *
from utils.data_utils import get_article_text


class NewsDataset(Dataset):

    def __init__(self, datalist: list, lang: str = "en", directory: str = DATA_PATH):
        """
        Arguments:
        datalist: a list object instance returned by the `select_universal_files` function.
        lang (str): The desired language to use.
        The valid options are the keys of LANG_DICT in `constants.py`.
        directory (str): The directory to the data folders.
        """

        self.data = datalist
        self.lang = lang
        self.directory = os.path.join(directory, lang)

    def get_lang(self):

        return self.lang

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        article_dir = os.path.join(self.directory, self.data[idx])
        text = get_article_text(article_dir, remove_p=True)

        return text, 0  # For now I'll assume real article = 0


def select_universal_files(langs: list = ["en"], directory: str = DATA_PATH):
    """
    Function to get every filename that exists in all the selected language directories.

    Arguments:
    langs (list): A list of languages to select.
    The valid options are the keys of LANG_DICT in `constants.py`.
    directory (str): The directory to the data folders.

    Returns:
    The list of filenames that exist in all the listed language folders.
    """

    # Get the list of subdirectories in the parent directory
    subdirs = [
        d
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d)) and d in langs
    ]

    # Get the files in each subdirectory
    file_sets = []
    for subdir in subdirs:

        folder_path = os.path.join(directory, subdir)
        file_sets.append(set(os.listdir(folder_path)))

    # Get the intersection
    common_files = list(set.intersection(*file_sets))

    return common_files
