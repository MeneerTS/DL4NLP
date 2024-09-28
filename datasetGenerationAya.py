import os, torch, argparse, json
from tqdm import tqdm
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.dataUtils import (
    count_tokens_in_document,
    get_article_text,
    extract_title_and_sentence,
)

with open("token.json") as f:
    token = json.load(f)["token"]
    login(token=token)


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        default="CohereForAI/aya-23-8B",
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
        "--languages",
        default=["en", "id", "zh", "de", "ru"],
        required=True,
        type=str.lower,
        nargs="+",
        help="The desired languages to generate for",
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


def generate_text(args):
    # Load the PolyLM-13B model and tokenizer
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Loop over the languages
    for language in args.languages:
        # Define the source and destination directories
        source_dir = f"{args.target_folder}/human/{language}_files"
        destination_dir = f"{args.target_folder}/aya/{language}_files"
        os.makedirs(destination_dir, exist_ok=True)

        # Loop through all .txt files in the source directory
        for file_name in tqdm(
            os.listdir(source_dir), desc="Processing files", unit="file"
        ):
            if file_name.endswith(".txt"):
                file_path = os.path.join(source_dir, file_name)

                # Read the content of the file
                text = get_article_text(file_path, remove_n=False)
                article_length = count_tokens_in_document(
                    text, language, use_period=False
                )

                # Extract the title and the first sentence
                title, sentence = extract_title_and_sentence(text)
                print(title)
                print(sentence)

                # Define the prompt for text generation
                if language == "en":
                    input_doc = (
                        f"Write a news article with the following headline: '{title}'. "
                        f"Start your article with the following sentence: '{sentence}'. "
                        f"Do not include separate headlines in the article. "
                        f"The article must contain a maximum of 100 words more or less than {article_length}."
                    )

                elif language == "de":
                    input_doc = (
                        f"Schreiben sie einen nachrichtenartikel mit der folgenden überschrift: '{title}'."
                        f"Beginnen sie ihren artikel mit folgendem satz: '{sentence}'."
                        "Fügen sie keine separaten überschriften in den artikel ein."
                        f"Der artikel darf höchstens 100 wörter mehr oder weniger als {article_length} enthalten.",
                    )

                elif language == "id":
                    input_doc = (
                        f"Tulis satu artikel berita dengan judul sebagai berikut: '{title}'."
                        f"Mulai artikel anda dengan kalimat berikut ini: '{sentence}'."
                        "Jangan sertakan judul terpisah dalam artikelnya."
                        f"Artikelnya harus mengandung maksimal 100 kata lebih atau kurang dari {article_length}.",
                    )

                elif language == "ru":
                    input_doc = (
                        f"Напишите новостную статью с заголовком: '{title}'."
                        f"Начните ваше эссе со следующего предложения: '{sentence}'."
                        "Не включайте отдельные заголовки в статью."
                        f"Статья должна содержать максимум 100 слов больше или меньше, чем {article_length}.",
                    )

                elif language == "zh":
                    input_doc = (
                        f"写一篇新闻报道，标题如下：'{title}'。"
                        f"用以下句子作为文章的开头 '{sentence}'。"
                        "文章中不要包含单独的标题。"
                        f"文章字数不得超过或少于{article_length}，最多 100 字。",
                    )

                messages = [
                    {
                        "role": "user",
                        "content": input_doc,
                    }
                ]

                # Tokenize the input
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(device)

                # Generate output
                with torch.no_grad():
                    generate_ids = model.generate(
                        input_ids,
                        do_sample=True,
                        max_length=args.max_length,
                        temperature=args.temperature,
                        top_p=0.9,
                    )

                # Decode the generated article
                generated_text = tokenizer.decode(
                    generate_ids[0], skip_special_tokens=True
                )

                # Save the generated article in the destination folder
                output_file_path = os.path.join(destination_dir, file_name)
                with open(output_file_path, "w", encoding="utf-8") as output_file:
                    output_file.write(f"{title}\n\n{generated_text}")

    print(
        f"Article generation complete! Files saved in 'dataset/aya/{args.languages}_files'."
    )


if __name__ == "__main__":
    args = config()
    generate_text(args)