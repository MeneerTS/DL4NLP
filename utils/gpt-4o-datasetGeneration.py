import torch, os, re
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)

languages = ['en', 'zh', 'de', 'id', 'ru']

# Mapping from language codes to language-specific settings
language_settings = {
    'en': {
        'name': 'english',
        'prompt': lambda title, sentence, article_length: [
            {"role": "user", "content": f"Write a news article with the following headline in English: '{title}'. "
                                        f"Start your article with the following sentence: '{sentence}'. "
                                        "Do not print the title and do not include separate headlines in the article."
                                        f"The article should be approximately {article_length} words, with the maximum difference of 100 words."}
        ]
    },
    'zh': {
        'name': 'chinese',
        'prompt': lambda title, sentence, article_length: [
            {"role": "user", "content": f"用以下标题写一篇新闻文章: '{title}'。"
                                        f"用以下句子开始你的文章: '{sentence}'。"
                                        "不要打印标题，也不要在文章中包含单独的标题。"
                                        f"文章的长度应大约为 {article_length} 字符，最大差异为 50 字符。"}
        ]
    },
    'de': {
        'name': 'german',
        'prompt': lambda title, sentence, article_length: [
            {"role": "user", "content": f"Schreiben Sie einen Nachrichtenartikel mit der folgenden Überschrift: '{title}'. "
                                        f"Beginnen Sie Ihren Artikel mit folgendem Satz: '{sentence}'. "
                                        "Drucken Sie den Titel nicht und fügen Sie keine separaten Überschriften in den Artikel ein."
                                        f"Der Artikel sollte ungefähr {article_length} Wörter lang sein, mit einer maximalen Abweichung von 100 Wörtern."}
        ]
    },
    'id': {
        'name': 'indonesian',
        'prompt': lambda title, sentence, article_length: [
            {"role": "user", "content": f"Tulislah artikel berita dengan judul berikut: '{title}'. "
                                        f"Mulailah artikel Anda dengan kalimat berikut: '{sentence}'. "
                                        "Jangan mencetak judul dan jangan menyertakan judul terpisah di dalam artikel."
                                        f"Artikel tersebut harus memiliki panjang sekitar {article_length} kata, dengan perbedaan maksimal 100 kata."}
        ]
    },
    'ru': {
        'name': 'russian',
        'prompt': lambda title, sentence, article_length: [
            {"role": "user", "content": f"Напишите новостную статью со следующим заголовком: '{title}'. "
                                        f"Начните свою статью со следующего предложения: '{sentence}'. "
                                        "Не печатайте заголовок и не включайте отдельные подзаголовки в статью."
                                        f"Статья должна быть приблизительно {article_length} слов, с максимальной разницей в 100 слов."}
        ]
    },
}

# Loop over each language
for language in languages:
    lang_name = language_settings[language]['name']

    home = str(Path.home())
    source_dir = os.path.join(home, f"dataset/human/{language}_files")
    destination_dir = os.path.join(home, f"dataset/machine/gpt-4o-mini/{language}_files")

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    def extract_title_and_sentence(text):
        # Split the text by lines
        lines = text.strip().split('\n')
        
        # The title is the first non-empty line
        title = lines[0].strip() if lines else "No Title Found"
        
        # Find the first sentence in the remaining text
        remaining_text = ' '.join(lines[1:]).strip()  # Join everything after the title

        if language == 'zh':
            sentence_match = re.search(r'([^。！？]*[。！？])', remaining_text) # Typical Chinese punctuation
        
        else:
            sentence_match = re.search(r'([^.]*?\.)', remaining_text)
        
        
        # Get the first sentence or a default message if not found
        sentence = sentence_match.group(1).strip() if sentence_match else "No sentence found"
        
        return title, sentence

    # Loop through all .txt files in the source directory
    for file_name in tqdm(os.listdir(source_dir), desc=f"Processing files for {lang_name}", unit="file"):
        if file_name.endswith(".txt"):
            file_path = os.path.join(source_dir, file_name)

            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            if language == 'zh':
                # Count n of characters (without blank spaces) for the Chinese prompt
                article_length = len(text.replace(" ", "").replace("\n", "").replace("\t", "")) 

            else:
                article_length= len(text.split())

            title, sentence = extract_title_and_sentence(text)

            messages = language_settings[language]['prompt'](title, sentence, article_length)

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            
            output_file_path = os.path.join(destination_dir, file_name)
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(f"{title}\n\n{completion.choices[0].message.content}")

    print(f"Article generation complete for {lang_name}. Files saved in 'dataset/machine/gpt-4o-mini/{language}_files'.")
            