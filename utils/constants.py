# File to store constant variables
import os
from lingua import Language

# For downloading the data
DATA_LINK = "http://data.statmt.org/news-commentary/v16/documents.tgz"

# For language checking
LANG_DICT = {
    "ar": Language.ARABIC,
    "cs": Language.CZECH,
    "de": Language.GERMAN,
    "en": Language.ENGLISH,
    "es": Language.SPANISH,
    "fr": Language.FRENCH,
    "hi": Language.HINDI,
    "id": Language.INDONESIAN,
    "it": Language.ITALIAN,
    "ja": Language.JAPANESE,
    "kk": Language.KAZAKH,
    "nl": Language.DUTCH,
    "pt": Language.PORTUGUESE,
    "ru": Language.RUSSIAN,
    "zh": Language.CHINESE,
}

# For establishing languages that are closely related enough (to allow for a bit more leniency during
# text cleaning, as the detector often mistakes these languages)
LANG_REL = {
    "ar": [Language.PERSIAN],
    "cs": [Language.SLOVAK],
    "de": [],
    "en": [],
    "es": [],
    "fr": [],
    "hi": [Language.BENGALI, Language.MARATHI],
    "id": [Language.MALAY],
    "it": [],
    "ja": [],
    "kk": [],
    "nl": [],
    "pt": [],
    "ru": [
        Language.MONGOLIAN,
        Language.SERBIAN,
        Language.MACEDONIAN,
        Language.BULGARIAN,
        Language.BELARUSIAN,
        Language.KAZAKH,
    ],
    "zh": [],
}

# For files
DATA_PATH = "data"
SPLIT_PATH = os.path.join(DATA_PATH, "split")
TAR_PATH = os.path.join(DATA_PATH, "documents.tgz")
DONE_DIR = os.path.join(DATA_PATH, "done.txt")
QWEN_PATH = "qwen"
MISTRAL_PATH = "mistral"