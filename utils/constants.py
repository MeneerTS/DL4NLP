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

# For files
DATA_PATH = "data"
SPLIT_PATH = os.path.join(DATA_PATH, "split")
TAR_PATH = os.path.join(DATA_PATH, "documents.tgz")
