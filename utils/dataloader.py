import os, re
from torch.utils.data import Dataset
from utils.constants import *
from utils.dataUtils import get_article_text


class DetectionDataset(Dataset):

    def __init__(
        self,
        language: str.lower = "en",
        human_source: str = HUMAN_PATH,
        ai_source: str = MISTRAL_PATH,
        n_samples: int = 50,
        sentence_mode: bool = False,
        n_sentences: int = 1,
    ):
        """
        Arguments:
        language (str): The desired language to use.
        The valid options are the keys of LANG_DICT in `constants.py`.
        human_source (str): Where to get the human-written files.
        ai_source (str): The folder name of the AI data to use.
        n_samples (int): The number of samples to take per half.
        sentence_mode (bool): If true, only take a sample sentence from the text.
        n_sentences (int): The number of sentences to sample (will be clipped if none exist).
        Note: the above parameter is especially important when handling files for DetectGPT,
        as it samples the text for 30 tokens (we do not want it to sample the original sentence).
        """
        # Check if n_samples if positive
        assert n_samples > 0, "Please input a positive value for `n_samples`!"

        # Get the filenames from the reference (human) folder
        datalist = os.listdir(os.path.join(human_source, f"{language}_files"))

        # Generate the full datalist and labels based on the folder locations
        # (i.e., if in the "human" folder, return label 0, else 1)
        real_path_base = os.path.join(human_source, f"{language}_files")
        ai_path_base = os.path.join(ai_source, f"{language}_files")

        self.real_paths, self.ai_paths, self.data = [], [], []
        for filename in datalist[:n_samples]:

            real_path_full = os.path.join(real_path_base, filename)
            ai_path_full = os.path.join(ai_path_base, filename)

            # For the DetectGPT experiments
            self.real_paths.append(real_path_full)
            self.ai_paths.append(ai_path_full)

            # For the standard dataloader
            self.data.extend([(real_path_full, 0), (ai_path_full, 1)])

        self.language = language
        self.sentence_mode = sentence_mode
        self.n_sentences = n_sentences

    def get_language(self):

        return self.language

    def get_filepaths(self):

        return self.data

    def get_detect_gpt_data(self):

        real_texts = [
            get_article_text(path, remove_h=False, remove_p=True, remove_n=False)
            for path in self.real_paths
        ]
        ai_texts = [
            get_article_text(path, remove_h=False, remove_p=True, remove_n=False)
            for path in self.ai_paths
        ]

        if self.sentence_mode:

            real_texts = [
                get_sample_sentence(text, self.n_sentences, self.language)
                for text in real_texts
            ]
            ai_texts = [
                get_sample_sentence(text, self.n_sentences, self.language)
                for text in ai_texts
            ]

        detect_gpt_data = {"original": real_texts, "sampled": ai_texts}

        return detect_gpt_data

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        article_dir, label = self.data[idx]
        text = get_article_text(
            article_dir, remove_h=False, remove_p=True, remove_n=False
        )

        # Return sample sentence if enabled
        if self.sentence_mode:
            text = get_sample_sentence(text, self.n_sentences, self.language)

        return text, label


# For getting only a sentence
def get_sample_sentence(text: str, n_sentences: int = 1, language: str = "en"):
    """
    Function to sample sentences from a text.
    Arguments:
    text (str): The text to sample from.
    n_sentences (int): The number of sentences to sample (will be clipped if fewer exist).
    language (str): The language of the text (default is "en" for English).

    Returns:
    A list of sampled sentences.
    """
    if language.lower() == "zh":
        sentence_pattern = SENTENCE_PATTERN_ZH

    else:
        sentence_pattern = SENTENCE_PATTERN_EN

    # Find all sentences (from the whole text without splitting/joining lines)
    sentences = sentence_pattern.findall(text)

    # Return the requested number of sentences, excluding the first one
    return "\n".join(sentences[1 : n_sentences + 1])


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
