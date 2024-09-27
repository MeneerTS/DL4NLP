import os, torch, argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.dataUtils import count_tokens_in_document, get_article_text


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        default="DAMO-NLP-MT/polylm-13b",
        required=True,
        type=str,
        help="The LLM to use for generation",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        required=True,
        type=str,
        help="Which device to use",
    )
    parser.add_argument(
        "--language",
        default="en",
        required=True,
        type=str.lower,
        help="The desired language to generate for",
    )
    parser.add_argument(
        "--source_dir",
        default="",
        required=True,
        type=str,
        help="The HuggingFace token",
    )
    parser.add_argument(
        "--max_length",
        default=2000,
        required=True,
        type=int,
        help="The max length of the model output per prompt in tokens",
    )
    parser.add_argument(
        "--temperature",
        default=0.6,
        required=True,
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


# This is for the cleaned files
def extract_title_and_sentence(text):
    sentences = text.split("\n")
    title, sentence = sentences[0], sentences[1]
    return title, sentence


def generate_text(args):
    # Load the PolyLM-13B model and tokenizer
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)

    # Define the source and destination directories
    source_dir = f"../../dataset/human/{args.language}_files"
    destination_dir = f"../../dataset/polylm/{args.language}_files"
    os.makedirs(destination_dir, exist_ok=True)

    # Loop through all .txt files in the source directory
    for file_name in tqdm(os.listdir(source_dir), desc="Processing files", unit="file"):
        if file_name.endswith(".txt"):
            file_path = os.path.join(source_dir, file_name)

            # Read the content of the file
            text = get_article_text(file_path)
            article_length = count_tokens_in_document(
                text, args.language, use_period=False
            )

            # Extract the title and the first sentence
            title, sentence = extract_title_and_sentence(text)

            # Define the prompt for text generation
            if args.language == "en":
                input_doc = (
                    f"Write a news article with the following headline: '{title}'. "
                    f"Start your article with the following sentence: '{sentence}'. "
                    f"Do not include separate headlines in the article. "
                    f"The article must contain a maximum of 100 words more or less than {article_length}."
                )
            elif args.language == "ar":
                input_doc = (
                    f".'{title}': اكتب مقالاً إخبارياً بالعنوان التالي"
                    f".'{sentence}' :ابدأ مقالك بالجملة التالية"
                    ".لا تدرج عناوين منفصلة في المقال"
                    f".{article_length} يجب أن تحتوي المقالة على 100 كلمة كحد أقصى أكثر أو أقل من",
                )

            elif args.language == "cs":
                input_doc = (
                    f"Napište článek s následujícím titulkem: '{title}'."
                    f"Začněte článek následující větou: '{sentence}'."
                    "Do článku nezařazujte samostatné titulky."
                    f"Článek musí obsahovat maximálně o 100 slov více nebo méně než {article_length}.",
                )

            elif args.language == "de":
                input_doc = (
                    f"Schreiben sie einen nachrichtenartikel mit der folgenden überschrift: '{title}'."
                    f"Beginnen sie ihren artikel mit folgendem satz: '{sentence}'."
                    "Fügen sie keine separaten überschriften in den artikel ein."
                    f"Der artikel darf höchstens 100 wörter mehr oder weniger als {article_length} enthalten.",
                )

            elif args.language == "es":
                input_doc = (
                    f"Escribe una noticia con el siguiente titular: '{title}'."
                    f"Comience el artículo con la siguiente frase: '{sentence}'."
                    "No incluya titulares separados en el artículo."
                    f"El artículo debe contener un máximo de 100 palabras más o menos que {article_length}.",
                )

            elif args.language == "fr":
                input_doc = (
                    f"Rédigez un article avec le titre suivant: '{title}'."
                    f"Commencez votre article par la phrase suivante: '{sentence}'."
                    "N'incluez pas de titres distincts dans l'article."
                    f"L'article doit contenir au maximum 100 mots de plus ou de moins que {article_length}.",
                )

            elif args.language == "hi":
                input_doc = (
                    f"निम्नलिखित शीर्षक के साथ एक समाचार लेख लिखें: '{title}'."
                    f"अपने लेख की शुरुआत निम्नलिखित वाक्य से करें: '{sentence}'."
                    "लेख में अलग-अलग सुर्खियाँ शामिल न करें।."
                    f"लेख में अधिकतम 100 शब्द {article_length} से अधिक या कम होने चाहिए।.",
                )

            elif args.language == "id":
                input_doc = (
                    f"Tulis satu artikel berita dengan judul sebagai berikut: '{title}'."
                    f"Mulai artikel anda dengan kalimat berikut ini: '{sentence}'."
                    "Jangan sertakan judul terpisah dalam artikelnya."
                    f"Artikelnya harus mengandung maksimal 100 kata lebih atau kurang dari {article_length}.",
                )

            elif args.language == "it":
                input_doc = (
                    f"Scrivi un articolo di notizie con il seguente titolo: '{title}'."
                    f"Inizia il tuo articolo con la seguente frase: '{sentence}'."
                    "Non includere titoli separati nell'articolo."
                    f"L'articolo deve contenere un massimo di 100 parole in più o in meno rispetto a {article_length}.",
                )

            elif args.language == "ja":
                input_doc = (
                    f"次のタイトルでニュース記事を書いてください： '{title}'。"
                    f"次の文章で記事を書き始めなさい： '{sentence}'。"
                    "記事に別の見出しをつけないでください。"
                    f"記事の文字数は最大100字以上{article_length}未満とします。",
                )

            elif args.language == "kk":
                input_doc = (
                    f"Келесі тақырыппен жаңалық мақаласын жазыңыз: '{title}'."
                    f"Эссені келесі сөйлеммен бастаңыз: '{sentence}'."
                    "Мақалаға бөлек тақырыптарды қоспаңыз."
                    f"Мақалада {article_length} мәнінен кем немесе көп 100 сөз болуы керек.",
                )

            elif args.language == "nl":
                input_doc = (
                    f"Schijf een nieuwsartikel met de volgende titel: '{title}'."
                    f"Begin je artikel met de volgende zin: '{sentence}'"
                    "Voeg geen aparte koppen toe aan het artikel."
                    f"Het artikel moet maximal 100 meer of minder dan {article_length} woorden bevatten.",
                )

            elif args.language == "pt":
                input_doc = (
                    f"Escreva um artigo de notícias com o seguinte título: '{title}'."
                    f"Comece o seu artigo com a seguinte frase: '{sentence}'."
                    "Não inclua títulos separados no artigo."
                    f"O artigo deve conter um máximo de 100 palavras a mais ou a menos do que {article_length}.",
                )

            elif args.language == "ru":
                input_doc = (
                    f"Напишите новостную статью с заголовком: '{title}'."
                    f"Начните ваше эссе со следующего предложения: '{sentence}'."
                    "Не включайте отдельные заголовки в статью."
                    f"Статья должна содержать максимум 100 слов больше или меньше, чем {article_length}.",
                )

            elif args.language == "zh":
                input_doc = (
                    f"写一篇新闻报道，标题如下：'{title}'。"
                    f"用以下句子作为文章的开头 '{sentence}'。"
                    "文章中不要包含单独的标题。"
                    f"文章字数不得超过或少于{article_length}，最多 100 字。",
                )

            # Tokenize the input
            inputs = tokenizer(input_doc, return_tensors="pt").to(device)

            # Generate output
            with torch.no_grad():
                generate_ids = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    do_sample=True,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_p=0.9,
                    num_beams=4,  # More diverse generation with beam search
                    early_stopping=True,
                )

            # Decode the generated article
            generated_text = tokenizer.decode(generate_ids[0], skip_special_tokens=True)

            # Save the generated article in the destination folder
            output_file_path = os.path.join(destination_dir, file_name)
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.write(f"{title}\n\n{generated_text}")

    print(
        f"Article generation complete! Files saved in 'dataset/machine/{args.language}_files'."
    )


if __name__ == "__main__":
    args = config()
    generate_text(args)
