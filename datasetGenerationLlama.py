import os, torch, argparse, re
from tqdm import tqdm
from transformers import pipeline, LlamaForCausalLM, AutoTokenizer
from utils.setSeed import set_seed_all
from utils.promptUtils import create_prompt
from dotenv import load_dotenv
from utils.cleanGeneratedText import clean_llama_articles
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
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
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
        help="The HuggingFace token",
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

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    # Set up the pipeline with the Hugging Face token (if needed)
    tokenizer = AutoTokenizer.from_pretrained(
    args.model_id, token=token
    )
    tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline(
        "text-generation",
        model=args.model_id,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        # model_kwargs={"torch_dtype": torch.float16}, 
        device=args.device,
        token=token,
    )
    # model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B", token=token)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)

    for language in args.languages: 
    # Define the source and destination directories
        source_dir = f"{args.target_folder}/human/{language}_files"
        destination_dir = f"{args.target_folder}/machine/llama-3.1-8B/{language}_files"
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

                # Create messages for the system role
                system_message = {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                }

                user_message = {"role": "user", "content": user_prompt}
                messages = [system_message, user_message]

                # chat_input = tokenizer.apply_chat_template(
                #     messages, tokenize=False, add_generation_prompt=True
                # )

                # model_inputs = tokenizer([chat_input], return_tensors="pt").to(device)

                # Generate output using the model
                # with torch.no_grad():
                #     generated_ids = model.generate(
                #         **model_inputs,
                #         max_new_tokens=args.max_length,
                #         do_sample=True,
                #         temperature=args.temperature,
                #         top_p=0.9,
                #     )

                # Extract only the newly generated tokens (ignore input tokens)
                # generated_ids = [
                #     output_ids[len(input_ids) :]
                #     for input_ids, output_ids in zip(
                #         model_inputs.input_ids, generated_ids
                #     )
                # ]

                 # Decode the generated text
                # generated_text = tokenizer.batch_decode(
                #     generated_ids, skip_special_tokens=True
                # )[0]
                
                # Define terminators (EOS tokens)
                terminators = [
                    pipe.tokenizer.eos_token_id,
                    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]

                # Generate the article using the model
                outputs = pipe(
                    messages,
                    max_new_tokens=args.max_length,
                    # eos_token_id=terminators,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.9,
                )

                # Extract the generated text
                assistant_response = outputs[0]["generated_text"][-1]["content"]

                # Save the generated article in the destination folder
                output_file_path = os.path.join(destination_dir, file_name)
                # print(assistant_response)
                # break
                with open(output_file_path, "w", encoding="utf-8") as output_file:
                    output_file.write(f"{title}\n\n{assistant_response}")
                    # output_file.write(f"{title}\n\n{generated_text}")

        print(
            f"Article generation complete! Files saved in '{args.target_folder}/machine/llama-3.1-8B/{language}_files'."
        )


if __name__ == "__main__":

    args = config()
    set_seed_all(args.seed)
    generate_text(args)

    print("Cleaning files...")
    clean_llama_articles(args.languages, f"{args.target_folder}/machine/llama-3.1-8B")
    print("Cleaning done!\n")
