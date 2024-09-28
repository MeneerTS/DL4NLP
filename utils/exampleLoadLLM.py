# By Thijs; Tiny model for testing:
import torch
from transformers import LlamaForCausalLM, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
# model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
# tokenizer.pad_token = tokenizer.eos_token #THIS IS HACKY DONT USE IT WITH LLAMA PLZ
hf_token = ".."

# Actual model:
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B", token=hf_token
)
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B", token=hf_token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Here we have to load in our input sentences instead
# Probably in a loop since we can't have infinite batch size
prompts = [
    "Het is een nacht die je normaal alleen",
    "Je suis un baguette et tu est un",
    "Das ist gans geil,",
]

# return tensors makes sure we return a tensor that can be used as a batch easily
encodings = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
input_ids = encodings["input_ids"]
# attention_mask = encodings["attention_mask"]

# Stuff we can edit
output_sentences_tokenized = model.generate(
    input_ids=input_ids,
    # attention_mask=attention_mask,
    max_length=500,  # max_length of prompts
    num_return_sequences=1,
    temperature=0.7,  # Lower temperature for more coherent text
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

# Just for testing
for sentence in output_sentences_detokenized:
    print("Next:")
    print(sentence)
