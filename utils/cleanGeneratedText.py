import os
from utils.constants import *
from utils.dataUtils import get_article_text


# Base filter function
def filter_merge_sentences(sentences: list, remove_start: int = 2):

    if sentences == []:
        return "None"

    new_sents = list(filter(None, sentences))[remove_start:]
    new_sents = [
        sent
        for sent in new_sents
        if ("word count" not in sent.lower() and "---" not in sent.lower())
    ]

    joined = "\n".join(sentences)

    return joined


# Function to clean the articles generated by Mistral-Small-Instruct
def clean_mistral_articles(languages: list = ["en", "zh", "de", "id", "ru"]):

    for language in languages:

        source_dir = f"{MISTRAL_PATH}\\{language}_files"

        for file in os.listdir(source_dir):

            if file.endswith(".txt"):

                fullname = os.path.join(source_dir, file)
                text = get_article_text(fullname, remove_n=False)

                sentences = text.split("\n")
                title = sentences[0]

                # The first two sentences are just the title and prompt
                joined = filter_merge_sentences(sentences, 2)

                new_text = f"""{title}\n\n{joined}"""

                with open(fullname, "w+") as f:

                    f.write(new_text)


# Function to clean articles generated by Qwen-2.5-Instruct
def clean_qwen_articles(languages: list = ["en", "zh", "de", "id", "ru"]):

    for language in languages:

        source_dir = f"{QWEN_PATH}\\{language}_files"

        for file in os.listdir(source_dir):

            if file.endswith(".txt"):

                fullname = os.path.join(source_dir, file)
                text = get_article_text(fullname, remove_n=False)

                sentences = text.split("\n")
                title = sentences[0]

                # The first two sentences are just the title and chat history
                joined = filter_merge_sentences(sentences, 6)

                new_text = f"""{title}\n\n{joined}"""

                with open(fullname, "w+") as f:

                    f.write(new_text)
