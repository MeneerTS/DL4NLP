import os
import re
from transformers import pipeline
import torch
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('HF_TOKEN')

# Set up the pipeline (add your Hugging Face token if required)
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
    token=token
)

# List of languages to process
languages = ['en', 'zh', 'de', 'id', 'ru']

# Mapping from language codes to language-specific settings
language_settings = {
    'en': {
        'name': 'english',
        'prompt': lambda title, sentence, article_length: [
            {"role": "user", "content": f"Write a news article with the following headline in English: '{title}'. "
                                        f"Start your article with the following sentence: '{sentence}'. "
                                        "Do not include separate headlines in the article. "
                                        f"The article should be approximately {article_length} words, with the maximum difference of 100 words."}
        ]
    },
    'zh': {
        'name': 'chinese',
        'prompt': lambda title, sentence, article_length: [
            {"role": "user", "content": f"用以下标题写一篇新闻文章: '{title}'。"
                                        f"用以下句子开始你的文章: '{sentence}'。"
                                        "不要在文章中包含单独的标题。"
                                        f"文章的长度应大约为 {article_length} 字，最大差异为 100 字。"}
        ]
    },
    'de': {
        'name': 'german',
        'prompt': lambda title, sentence, article_length: [
            {"role": "user", "content": f"Schreiben Sie einen Nachrichtenartikel mit der folgenden Überschrift: '{title}'. "
                                        f"Beginnen Sie Ihren Artikel mit folgendem Satz: '{sentence}'. "
                                        "Fügen Sie keine separaten Überschriften in den Artikel ein. "
                                        f"Der Artikel sollte ungefähr {article_length} Wörter lang sein, mit einer maximalen Abweichung von 100 Wörtern."}
        ]
    },
    'id': {
        'name': 'indonesian',
        'prompt': lambda title, sentence, article_length: [
            {"role": "user", "content": f"Tulislah artikel berita dengan judul berikut: '{title}'. "
                                        f"Mulailah artikel Anda dengan kalimat berikut: '{sentence}'. "
                                        "Jangan sertakan judul terpisah dalam artikel. "
                                        f"Artikel tersebut harus memiliki panjang sekitar {article_length} kata, dengan perbedaan maksimal 100 kata."}
        ]
    },
    'ru': {
        'name': 'russian',
        'prompt': lambda title, sentence, article_length: [
            {"role": "user", "content": f"Напишите новостную статью со следующим заголовком: '{title}'. "
                                        f"Начните свою статью со следующего предложения: '{sentence}'. "
                                        "Не включайте отдельные заголовки в статью. "
                                        f"Статья должна быть приблизительно {article_length} слов, с максимальной разницей в 100 слов."}
        ]
    },
}

# Loop over each language
for language in languages:
    lang_name = language_settings[language]['name']
    
    # Define the source and destination directories
    home = str(Path.home())
    source_dir = os.path.join(home, f"dataset/human/{language}_files")
    destination_dir = os.path.join(home, f"dataset/machine/{language}_files")

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Function to extract the title and first sentence from the text
    def extract_title_and_sentence(text):
        # Split the text by lines
        lines = text.strip().split('\n')
        
        # The title is the first non-empty line
        title = lines[0].strip() if lines else "No Title Found"
        
        # Find the first sentence in the remaining text
        remaining_text = ' '.join(lines[1:]).strip()  # Join everything after the title
        sentence_match = re.search(r'([^.]*?\.)', remaining_text)
        
        # Get the first sentence or a default message if not found
        sentence = sentence_match.group(1).strip() if sentence_match else "No Sentence Found"
        
        return title, sentence

    # Loop through all .txt files in the source directory
    for file_name in tqdm(os.listdir(source_dir), desc=f"Processing files for {lang_name}", unit="file"):
        if file_name.endswith(".txt"):
            file_path = os.path.join(source_dir, file_name)

            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            article_length = len(text.split())

            # Extract the title and the first sentence
            title, sentence = extract_title_and_sentence(text)

            # Get the prompt messages for the current language
            messages = language_settings[language]['prompt'](title, sentence, article_length)

            # Define terminators (EOS tokens)
            terminators = [
                pipe.tokenizer.eos_token_id,
                pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            # Generate the article using the model
            outputs = pipe(
                messages,
                max_new_tokens=2000,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

            # Extract the generated text of the assistant
            assistant_response = outputs[0]["generated_text"][1]['content']

            # Save the generated article in the destination folder
            output_file_path = os.path.join(destination_dir, file_name)
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(f"{title}\n\n{assistant_response}")

    print(f"Article generation complete for {lang_name}. Files saved in 'dataset/machine/{language}_files'.")

