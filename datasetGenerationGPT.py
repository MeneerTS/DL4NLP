import os, torch, argparse, re
from tqdm import tqdm
from transformers import pipeline
from utils.setSeed import set_seed_all
from dotenv import load_dotenv
from openai import OpenAI
from utils.cleanGeneratedText import clean_gpt_articles
from utils.dataUtils import (
    count_tokens_in_document,
    get_article_text,
    # extract_title_and_sentence,
)


def config():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="The seed for reproductibility",
    )
    parser.add_argument(
        "--model_id",
        default="gpt-4o-mini",
        type=str,
        help="The LLM to use for generation",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Which device to use",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        help="The HuggingFace token",
    )
    parser.add_argument(
        "--language",
        default="en",
        type=str.lower,
        help="The desired language to generate for",
    )
    parser.add_argument(
        "--source_dir",
        default="",
        type=str,
        help="The source directory",
    )
    parser.add_argument(
        "--max_length",
        default=4000,
        type=int,
        help="The max length of the model output per prompt in tokens",
    )
    parser.add_argument(
        "--temperature",
        default=0.6,
        type=float,
        help="The model temperature for generation",
    )
    parser.add_argument(
        "--target_folder",
        required=True,
        type=str,
        help="The folder to save the files in",
    )
    args = parser.parse_args()

    return args


def extract_title_and_sentence(text):
    # Split the text by lines
    lines = text.strip().split('\n')
    
    # The title is the first non-empty line
    title = lines[0].strip() if lines else "No Title Found"
    
    # Find the first sentence in the remaining text
    remaining_text = ' '.join(lines[1:]).strip()  # Join everything after the title

    if args.language == 'zh':
        sentence_match = re.search(r'([^。！？]*[。！？])', remaining_text) # Typical Chinese punctuation
    
    else:
        sentence_match = re.search(r'([^.]*?\.)', remaining_text)
    
    # Get the first sentence or a default message if not found
    sentence = sentence_match.group(1).strip() if sentence_match else "No sentence found"
    
    return title, sentence

def generate_text(args):

    # Set up the pipeline with the Hugging Face token (if needed)
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    client = OpenAI(api_key=api_key)

    # Define the source and destination directories
    source_dir = f"{args.target_folder}/human/{args.language}_files"
    destination_dir = f"{args.target_folder}/machine/{args.model_id}/{args.language}_files"
    os.makedirs(destination_dir, exist_ok=True)

    # Loop through all .txt files in the source directory
    for file_name in tqdm(os.listdir(source_dir), desc=f"Processing files for {args.language}", unit="file"):

        if file_name.endswith(".txt"):
            file_path = os.path.join(source_dir, file_name)

            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            if args.language == 'zh':
                # Count n of characters (without blank spaces) for the Chinese prompt
                article_length = len(text.replace(" ", "").replace("\n", "").replace("\t", "")) 
            else:
                article_length= len(text.split())

            # Extract the title and the first sentence
            title, sentence = extract_title_and_sentence(text)

            # Define the prompt for text generation

            if args.language == "en":
                messages = [
                    {
                        "role": "user",
                        "content": f"Write a news article with the following headline in English: '{title}'. "
                                f"Start your article with the following sentence: '{sentence}'. "
                                "Do not print the title and do not include separate headlines in the article."
                                f"The article should be approximately {article_length} words, with the maximum difference of 100 words."
                    },
                ]

            elif args.language == "ar":
                messages = [
                    {
                        "role": "user",
                        "content": f".'{title}': اكتب مقالاً إخبارياً بالعنوان التالي"
                        f".'{sentence}' :ابدأ مقالك بالجملة التالية"
                        ".لا تدرج عناوين منفصلة في المقال"
                        f".{article_length} يجب أن تحتوي المقالة على 100 كلمة كحد أقصى أكثر أو أقل من",
                    },
                ]

            elif args.language == "cs":
                messages = [
                    {
                        "role": "user",
                        "content": f"Napište článek s následujícím titulkem: '{title}'."
                        f"Začněte článek následující větou: '{sentence}'."
                        "Do článku nezařazujte samostatné titulky."
                        f"Článek musí obsahovat maximálně o 100 slov více nebo méně než {article_length}.",
                    },
                ]

            elif args.language == "de":
                messages = [
                    {
                        "role": "user",
                        "content": f"Schreiben Sie einen Nachrichtenartikel mit der folgenden Überschrift: '{title}'. "
                                        f"Beginnen Sie Ihren Artikel mit folgendem Satz: '{sentence}'. "
                                        "Drucken Sie den Titel nicht und fügen Sie keine separaten Überschriften in den Artikel ein."
                                        f"Der Artikel sollte ungefähr {article_length} Wörter lang sein, mit einer maximalen Abweichung von 100 Wörtern."
                                        }        
                ]

            elif args.language == "es":
                messages = [
                    {
                        "role": "user",
                        "content": f"Escribe una noticia con el siguiente titular: '{title}'."
                        f"Comience el artículo con la siguiente frase: '{sentence}'."
                        "No incluya titulares separados en el artículo."
                        f"El artículo debe contener un máximo de 100 palabras más o menos que {article_length}.",
                    },
                ]

            elif args.language == "fr":
                messages = [
                    {
                        "role": "user",
                        "content": f"Rédigez un article avec le titre suivant: '{title}'."
                        f"Commencez votre article par la phrase suivante: '{sentence}'."
                        "N'incluez pas de titres distincts dans l'article."
                        f"L'article doit contenir au maximum 100 mots de plus ou de moins que {article_length}.",
                    },
                ]

            elif args.language == "hi":
                messages = [
                    {
                        "role": "user",
                        "content": f"निम्नलिखित शीर्षक के साथ एक समाचार लेख लिखें: '{title}'."
                        f"अपने लेख की शुरुआत निम्नलिखित वाक्य से करें: '{sentence}'."
                        "लेख में अलग-अलग सुर्खियाँ शामिल न करें।."
                        f"लेख में अधिकतम 100 शब्द {article_length} से अधिक या कम होने चाहिए।.",
                    },
                ]

            elif args.language == "id":
                messages = [
                    {
                        "role": "user",
                        "content": f"Tulislah artikel berita dengan judul berikut: '{title}'. "
                                    f"Mulailah artikel Anda dengan kalimat berikut: '{sentence}'. "
                                    "Jangan mencetak judul dan jangan menyertakan judul terpisah di dalam artikel."
                                    f"Artikel tersebut harus memiliki panjang sekitar {article_length} kata, dengan perbedaan maksimal 100 kata."
                    },
                ]

            elif args.language == "it":
                messages = [
                    {
                        "role": "user",
                        "content": f"Scrivi un articolo di notizie con il seguente titolo: '{title}'."
                        f"Inizia il tuo articolo con la seguente frase: '{sentence}'."
                        "Non includere titoli separati nell'articolo."
                        f"L'articolo deve contenere un massimo di 100 parole in più o in meno rispetto a {article_length}.",
                    },
                ]

            elif args.language == "ja":
                messages = [
                    {
                        "role": "user",
                        "content": f"次のタイトルでニュース記事を書いてください： '{title}'。"
                        f"次の文章で記事を書き始めなさい： '{sentence}'。"
                        "記事に別の見出しをつけないでください。"
                        f"記事の文字数は最大100字以上{article_length}未満とします。",
                    },
                ]

            elif args.language == "kk":
                messages = [
                    {
                        "role": "user",
                        "content": f"Келесі тақырыппен жаңалық мақаласын жазыңыз: '{title}'."
                        f"Эссені келесі сөйлеммен бастаңыз: '{sentence}'."
                        "Мақалаға бөлек тақырыптарды қоспаңыз."
                        f"Мақалада {article_length} мәнінен кем немесе көп 100 сөз болуы керек.",
                    },
                ]

            elif args.language == "nl":
                messages = [
                    {
                        "role": "user",
                        "content": f"Schijf een nieuwsartikel met de volgende titel: '{title}'."
                        f"Begin je artikel met de volgende zin: '{sentence}'"
                        "Voeg geen aparte koppen toe aan het artikel."
                        f"Het artikel moet maximal 100 meer of minder dan {article_length} woorden bevatten.",
                    },
                ]

            elif args.language == "pt":
                messages = [
                    {
                        "role": "user",
                        "content": f"Escreva um artigo de notícias com o seguinte título: '{title}'."
                        f"Comece o seu artigo com a seguinte frase: '{sentence}'."
                        "Não inclua títulos separados no artigo."
                        f"O artigo deve conter um máximo de 100 palavras a mais ou a menos do que {article_length}.",
                    },
                ]

            elif args.language == "ru":
                messages = [
                    {
                        "role": "user",
                        "content": f"Напишите новостную статью со следующим заголовком: '{title}'. "
                                    f"Начните свою статью со следующего предложения: '{sentence}'. "
                                    "Не печатайте заголовок и не включайте отдельные подзаголовки в статью."
                                    f"Статья должна быть приблизительно {article_length} слов, с максимальной разницей в 100 слов."
                        },
                ]

            elif args.language == "zh":
                messages = [
                    {
                        "role": "user",
                        "content": f"用以下标题写一篇新闻文章: '{title}'。"
                                    f"用以下句子开始你的文章: '{sentence}'。"
                                    "不要打印标题，也不要在文章中包含单独的标题。"
                                    f"文章的长度应大约为 {article_length} 字符，最大差异为 50 字符。"
                    },
                ]

            # Generate the article using the model
            completion = client.chat.completions.create(
                model=args.model_id,
                messages=messages
            )

            # Extract the generated text
            assistant_response = completion.choices[0].message.content

            # Save the generated article in the destination folder
            output_file_path = os.path.join(destination_dir, file_name)

            with open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.write(f"{title}\n\n{assistant_response}")

    print(
        f"Article generation complete! Files saved in 'dataset/machine/{args.model_id}/{args.language}_files'."
    )


if __name__ == "__main__":

    args = config()
    set_seed_all(args.seed)
    generate_text(args)

    print("Cleaning files...")
    clean_gpt_articles(args.language, f"{args.target_folder}/machine/{args.model_id}")
    print("Cleaning done!\n")
