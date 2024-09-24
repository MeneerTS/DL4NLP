import os
import re
from transformers import pipeline
import torch
from tqdm import tqdm

# Set up the pipeline with the Hugging Face token
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
    token=".."  # Your HF token
)

language = 'nl'

# Define the source and destination directories
source_dir = f'../../dataset/human/{language}_files'
destination_dir = f'../../dataset/machine/{language}_files'

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Function to extract the title and first sentence from the text
def extract_title_and_sentence(text):
    # Extract title after <HEADLINE>
    title_match = re.search(r"<HEADLINE>(.*?)<P>", text, re.DOTALL)
    title = title_match.group(1).strip() if title_match else "No Title Found"
    
    # Extract first sentence after <P> (until a dot is found)
    sentence_match = re.search(r"<P>(.*?\.)", text, re.DOTALL)
    sentence = sentence_match.group(1).strip() if sentence_match else "No Sentence Found"
    
    return title, sentence

# Loop through all .txt files in the source directory
# for file_name in os.listdir(source_dir):
for file_name in tqdm(os.listdir(source_dir), desc="Processing files", unit="file"):
    if file_name.endswith(".txt"):
        file_path = os.path.join(source_dir, file_name)
        
        # Read the content of the file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        article_length = len(text.split())
        
        # Extract the title and the first sentence
        title, sentence = extract_title_and_sentence(text)
        
        # Define the prompt for text generation

        if language == 'en':
            messages = [
                {"role": "user", "content": f"Write a news article with the following headline in English: '{title}'."
                                            f"Start your article with the following sentence: '{sentence}'"
                                            "Do not include separate headlines in the article."
                                            f"The article has to be around {article_length} words."},
            ]

        elif language == 'nl':
            messages = [
                {"role": "user", "content": f"Schijf een nieuwsartikel met de volgende titel: '{title}'."
                                            f"Begin je artikel met de volgende zin: '{sentence}'"
                                            "Voeg geen aparte koppen toe aan het artikel"
                                            f"Het artikel moet maximal 100 meer of minder dan {article_length} woorden bevatten."},
            ]
        
        elif language == 'it':
            messages = [
                {"role": "user", "content": f"Scrivi un articolo di notizie con il seguente titolo: '{title}'."
                                        f"Inizia il tuo articolo con la seguente frase: '{sentence}'."
                                        "Non includere titoli separati nell'articolo."
                                        f"L'articolo deve essere di circa {article_length} parole."},
            ]
        
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
        
        # Extract the generated text
        assistant_response = outputs[0]["generated_text"][-1]["content"]

        # Save the generated article in the destination folder
        output_file_path = os.path.join(destination_dir, file_name)
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(f"{title}\n\n{assistant_response}")

print(f"Article generation complete. Files saved in 'dataset/machine/{language}_files'.")
