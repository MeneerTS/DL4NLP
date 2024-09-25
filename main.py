# STILL UNFINISHED, NEED TO IMPLEMENT FULL TRAINING LOOP
import torch, argparse
from transformers import LlamaForCausalLM, AutoTokenizer


def config():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
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
        "--hf_token",
        required=True,
        type=str,
        help="The HuggingFace token",
    )
    parser.add_argument(
        "--languages",
        default=["en", "ru", "zh", "id", "nl"],
        required=True,
        type=list,
        help="The desired languages to test",
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
        default=1000,
        required=True,
        type=int,
        help="The max length of the model output per prompt in tokens",
    )
    parser.add_argument(
        "--temperature",
        default=0.1,
        required=True,
        type=float,
        help="The model temperature for generation",
    )
    args = parser.parse_args()

    return args


def load_model(args):
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    # model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    # tokenizer.pad_token = tokenizer.eos_token #THIS IS HACKY DONT USE IT WITH LLAMA PLZ
    # Load the actual model:
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B", token=args.hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B", token=args.hf_token
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, tokenizer, device


def prompt_model(prompts, model, tokenizer, device, args):

    # Here we have to load in our input sentences instead
    # Probably in a loop since we can't have infinite batch size

    # return tensors makes sure we return a tensor that can be used as a batch easily
    encodings = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    input_ids = encodings["input_ids"]
    # attention_mask = encodings["attention_mask"]

    # Stuff we can edit
    output_sentences_tokenized = model.generate(
        input_ids=input_ids,
        # attention_mask=attention_mask,
        max_length=args.max_length,  # max_length of prompts
        num_return_sequences=1,
        temperature=args.temperature,  # Lower temperature for more coherent text
        top_k=50,  # Sample from top50 instead of low chance of something weird
        top_p=0.95,  # Nucleus sampling
        do_sample=True,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,  # Handle padding
    )

    # Now we should write this to disk
    # To build our dataset
    output_sentences_detokenized = [
        tokenizer.decode(output, skip_special_tokens=True)
        for output in output_sentences_tokenized
    ]

    return output_sentences_detokenized


if __name__ == "__main__":

    args = config()
    prompts = [
        "Het is een nacht die je normaal alleen",
        "Je suis un baguette et tu est un",
        "Das ist gans geil,",
    ]
    model, tokenizer, device = load_model(args)
    prompt_model(prompts, model, tokenizer, device, args)
