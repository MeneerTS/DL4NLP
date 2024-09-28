# **DL4NLP**

Repository for the Deep Learning for Natural Language Processing course group project.

## **Setup**

To run this code, first do the following:

```
git clone MeneerTS/DL4NLP
```

Afterwards, you need to install the dependencies:

```
conda env create -f environment.yml
```

Use the below instead if you want to install with CUDA compatibility:

```
conda env create -f environment_gpu.yml
```

## **Running**

The datasets in this project were generated manually using the following models:

1. [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
2. [openai/ChatGPT-4o-mini](https://openai.com/index/hello-gpt-4o/)
3. [mistralai/Mistral-Small-Instruct-2409](https://huggingface.co/mistralai/Mistral-Small-Instruct-2409)
4. [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)

All the above models are at least somewhat capable of generating articles in the following languages with the correct prompt:

1. English (🇬🇧)
2. German (🇩🇪)
3. Chinese (🇨🇳)
4. Indonesian (🇮🇩)
5. Russian (🇷🇺)

To replicate the process, run any of the following for the dataset(s) you want:

```
python datasetGenerationLlama.py
```

```
python datasetGenerationGPT.py
```

```
python datasetGenerationMistral.py \
    --model_id "mistralai/Mistral-Small-Instruct-2409" \
    --device "cuda" \
    --languages "en" "id" "zh" "de" "ru" \
    --max_length 2000 \
    --temperature 0.6 \
    --target_folder "" # Where you want the Mistral files to be located in
```

```
python datasetGenerationQwen.py \
    --model_id "Qwen/Qwen2.5-32B-Instruct" \
    --device "cuda" \
    --languages "en" "id" "zh" "de" "ru" \
    --max_length 2000 \
    --temperature 0.6 \
    --target_folder "" # Where you want the Qwen files to be located in
```

All the selected filenames can be found/loaded from the `articles.txt` file.

_Note: To generate articles using LLaMA, GPT, and Mistral, you need an API token (a paid one in the case of GPT)._
_Currently, the way it is setup is that you need to have a `token.json` file with the following format:_

```py
{"token": "hf_YOUR_TOKEN"}
```

_... or just the standard way of having it in your environment variable._
_Also, ensure that the folders containing the data are in the same directory as this README._


To evaluate the generated datasets, run the following:

```
# WIP
```
