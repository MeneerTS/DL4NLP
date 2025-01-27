import os, torch, argparse, json
from tqdm import tqdm
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.setSeed import set_seed_all
from utils.promptUtils import create_prompt
from utils.cleanGeneratedText import clean_qwen_articles
from utils.dataUtils import (
    count_tokens_in_document,
    get_article_text,
    extract_title_and_sentence,
)

token = os.getenv("HF_TOKEN")
if not token:
    with open("token.json", encoding="utf-8") as f:
        token = json.load(f)["token"]
        login(token=token)


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
        default="Qwen/Qwen2.5-32B-Instruct",
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
        "--languages",
        default=["en", "id", "zh", "de", "ru"],
        type=str.lower,
        nargs="+",
        help="The desired languages to generate for",
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


def generate_text(args):
    # Load the model and tokenizer
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)

    # Loop over languages
    for language in args.languages:
        source_dir = f"{args.target_folder}/human/{language}_files"
        destination_dir = f"{args.target_folder}/qwen/{language}_files"
        os.makedirs(destination_dir, exist_ok=True)

        # Process all files in the source directory
        for file_name in tqdm(
            os.listdir(source_dir), desc=f"Processing {language} files", unit="file"
        ):
            if file_name.endswith(".txt"):
                file_path = os.path.join(source_dir, file_name)

                # Read content
                text = get_article_text(file_path, remove_n=False)
                article_length = count_tokens_in_document(
                    text, language, use_period=False
                )

                # Extract title and first sentence
                title, sentence = extract_title_and_sentence(text, language=language)
                print(f"Title: {title}")
                print(f"Sentence: {sentence}")

                # Define the prompt based on language
                user_prompt = create_prompt(
                    title=title,
                    sentence=sentence,
                    article_length=article_length,
                    language=language,
                )

                user_message = [{"role": "user", "content": user_prompt}]

                # Prepare inputs for the model using the new template and method
                chat_input = tokenizer.apply_chat_template(
                    user_message, tokenize=False, add_generation_prompt=True
                )
                model_inputs = tokenizer([chat_input], return_tensors="pt").to(device)

                # Generate output using the model
                with torch.no_grad():
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=args.max_length,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=0.9,
                    )

                # Extract only the newly generated tokens (ignore input tokens)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(
                        model_inputs.input_ids, generated_ids
                    )
                ]

                # Decode the generated text
                generated_text = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

                # Save the generated article
                output_file_path = os.path.join(destination_dir, file_name)
                with open(output_file_path, "w", encoding="utf-8") as output_file:
                    output_file.write(f"{title}\n\n{generated_text}")

    print(f"Article generation complete! Files saved in '{args.target_folder}/qwen/'.")


if __name__ == "__main__":

    args = config()
    set_seed_all(args.seed)
    generate_text(args)

    print("Cleaning files...")
    clean_qwen_articles(args.languages)
    print("Cleaning done!\n")
