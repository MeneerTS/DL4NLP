import os
from utils.constants import *
from utils.dataUtils import get_article_text


# Base filter function
def filter_merge_sentences(sentences: list, remove_start: int = 2):

    filter_terms = [
        "user",
        "Maaf",
        "Mohon Maaf",
        "RecognitionException",
        "Baik",
        "Tolong",
        "Bitte ignorieren",
        "Entschuldigung, aber",
    ]
    if sentences[remove_start:] == []:
        return "None"

    new_sents = list(filter(None, sentences))[remove_start:]
    new_sents = [
        sent
        for sent in new_sents
        if (
            "word count" not in sent.lower()  # Remove counts
            and len(sent) > 18  # Remove very short (likely useless sentences)
            and "user"
            not in sent  # The Indonesian dataset frequently encounters these (and the below) issues
            and not any([term in sent for term in filter_terms])
        )
    ]

    joined = "\n".join(new_sents)

    return joined


# Function to clean the articles generated by Mistral-Small-Instruct
def clean_mistral_articles(languages: list = ["en", "zh", "de", "id", "ru"]):

    for language in languages:

        source_dir = os.path.join(MISTRAL_PATH, f"{language}_files")
        clean_dir = os.path.join(f"{MISTRAL_PATH}_clean", f"{language}_files")
        # Ensure the target directory exists
        os.makedirs(clean_dir, exist_ok=True)

        for file in os.listdir(source_dir):

            if file.endswith(".txt"):
                fullpath = os.path.join(source_dir, file)
                fc_path = os.path.join(clean_dir, file)
                text = get_article_text(fullpath, remove_n=False)
                sentences = text.split("\n")
                title = sentences[0]

                # The first two sentences are just the title and prompt
                joined = filter_merge_sentences(sentences, 2)
                new_text = f"""{title}\n\n{joined}"""

                with open(fc_path, "w+", encoding="utf-8") as f:
                    f.write(new_text)


# Function to clean articles generated by Qwen-2.5-Instruct
def clean_qwen_articles(languages: list = ["en", "zh", "de", "id", "ru"]):

    for language in languages:

        source_dir = os.path.join(QWEN_PATH, f"{language}_files")
        clean_dir = os.path.join(f"{QWEN_PATH}_clean", f"{language}_files")
        # Ensure the target directory exists
        os.makedirs(clean_dir, exist_ok=True)

        for file in os.listdir(source_dir):

            if file.endswith(".txt"):
                fullpath = os.path.join(source_dir, file)
                fc_path = os.path.join(clean_dir, file)
                text = get_article_text(fullpath, remove_n=False)
                sentences = text.split("\n")
                title = sentences[0]

                # The first two sentences are just the title and prompt
                joined = filter_merge_sentences(sentences, 6)
                new_text = f"""{title}\n\n{joined}"""

                with open(fc_path, "w+", encoding="utf-8") as f:
                    f.write(new_text)
