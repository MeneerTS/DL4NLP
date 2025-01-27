import os, torch, argparse, re
from tqdm import tqdm
from transformers import pipeline
from utils.setSeed import set_seed_all
from utils.promptUtils import create_prompt
from dotenv import load_dotenv
from openai import OpenAI
from utils.cleanGeneratedText import clean_gpt_articles
from utils.dataUtils import (
    count_tokens_in_document,
    get_article_text,
    # extract_title_and_sentence,
)
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    with open("token.json") as f:
        api_key = json.load(f)["token"]

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
        "--languages",
        default=["en", "id", "zh", "de", "ru"],
        type=str.lower,
        nargs="+",
        help="The desired languages to generate for",
    )
    parser.add_argument(
        "--source_dir",
        default="",
        type=str,
        help="The source directory",
    )
    parser.add_argument(
        "--target_folder",
        required=True,
        type=str,
        help="The folder to save the files in",
    )
    args = parser.parse_args()

    return args


def extract_title_and_sentence(text, language):
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

def generate_text(args):

    # Set up the pipeline with the Hugging Face token (if needed)
    

    client = OpenAI(api_key=api_key)

    # Define the source and destination directories
    for language in args.languages:
        source_dir = f"{args.target_folder}/human/{language}_files"
        destination_dir = f"{args.target_folder}/machine/{args.model_id}/{language}_files"
        os.makedirs(destination_dir, exist_ok=True)

        # Loop through all .txt files in the source directory
        for file_name in tqdm(os.listdir(source_dir), desc=f"Processing files for {language}", unit="file"):

            if file_name.endswith(".txt"):
                file_path = os.path.join(source_dir, file_name)

                # Read the content of the file
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()

                if language == 'zh':
                    # Count n of characters (without blank spaces) for the Chinese prompt
                    article_length = len(text.replace(" ", "").replace("\n", "").replace("\t", "")) 
                else:
                    article_length= len(text.split())

                # Extract the title and the first sentence
                title, sentence = extract_title_and_sentence(text, language)

                # Define the prompt for text generation
                user_prompt = create_prompt(
                    title=title,
                    sentence=sentence,
                    article_length=article_length,
                    language=language,
                )

                user_message = {"role": "user", "content": user_prompt}
                messages = [user_message]

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
            f"Article generation complete! Files saved in 'dataset/machine/{args.model_id}/{language}_files'."
        )


if __name__ == "__main__":

    args = config()
    set_seed_all(args.seed)
    generate_text(args)

    print("Cleaning files...")
    clean_gpt_articles(args.languages, f"{args.target_folder}/machine/{args.model_id}")
    print("Cleaning done!\n")
