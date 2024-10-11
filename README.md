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

### **Dataset Generation**

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

To replicate the process, we first need to generate and clean the dataset. This can be done by simply performing the following:

```
python get_base_data.py
```

Alternatively, you can run it from the notebook [here](notebooks/example.ipynb) (alongside an overview of the dataloader).
Afterwards, run any of the following for the dataset(s) you want:

```
python datasetGenerationLlama.py \
    --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --device "cuda" \
    --languages "en" "id" "zh" "de" "ru" \
    --max_length 4000 \
    --temperature 0.6 \
    --target_folder "" # Where you want the Llama files to be located in
```

```
python datasetGenerationGPT.py
    --model_id "gpt-4o-mini" \
    --languages "en" "id" "zh" "de" "ru" \
    --target_folder "" # Where you want the GPT files to be located in
```

```
python datasetGenerationMistral.py \
    --model_id "mistralai/Mistral-Small-Instruct-2409" \
    --device "cuda" \
    --languages "en" "id" "zh" "de" "ru" \
    --max_length 4000 \
    --temperature 0.6 \
    --target_folder "" # Where you want the Mistral files to be located in
```

```
python datasetGenerationQwen.py \
    --model_id "Qwen/Qwen2.5-32B-Instruct" \
    --device "cuda" \
    --languages "en" "id" "zh" "de" "ru" \
    --max_length 4000 \
    --temperature 0.6 \
    --target_folder "" # Where you want the Qwen files to be located in
```

Do note that this process takes a long time, so perhaps it is better to download the files [here](https://amsuni-my.sharepoint.com/:f:/g/personal/gregory_go_student_uva_nl/EnzVCZAaSUJDvB-F66VVW-8B_BrvOwz_XSEB5N1_CDBezA?e=gctad6).

All the selected filenames can be found/loaded from the `articles.txt` file.

_Note: To generate articles using LLaMA, GPT, and Mistral, you need an API token (a paid one in the case of GPT)._
_Currently, the way it is setup is that you need to have a `token.json` file with the following format:_

```py
{"token": "hf_YOUR_TOKEN"}
```

_... or just the standard way of having it in your environment variable._
_Also, ensure that the folders containing the data are in the same directory as this README._

### **Evaluation**

To evaluate the generated datasets, run the following:

```sh
# Adjust the variables based on which experiment you want to do 
python -u main.py \
    --save_dir "..." \
    --ai_location "$current_ai_location" \
    --sentence_mode True \
    --n_sentences 15 \
    --language "ru" \
    --base_model_name "ai-forever/mGPT" \
    --mask_filling_model_name "google/mt5-base" \
    --n_documents 25 \
    --cache_dir "..."
```

This script was adapted from the [DetectGPT Repository](https://github.com/eric-mitchell/detect-gpt), which can be cited through the following:

```
@misc{mitchell2023detectgpt,
    url = {https://arxiv.org/abs/2301.11305},
    author = {Mitchell, Eric and Lee, Yoonho and Khazatsky, Alexander and Manning, Christopher D. and Finn, Chelsea},
    title = {DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature},
    publisher = {arXiv},
    year = {2023},
}
```
