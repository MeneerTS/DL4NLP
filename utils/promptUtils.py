# Function to store prompts for all models


def create_prompt(
    title: str, sentence: str, article_length: int = 600, language: str.lower = "en"
):
    """
    Returns the corresponding prompt for each language

    Arguments:
    title (str): The title of the article to generate.
    sentence (str): The first sentence of the article to generate.
    article_length (int): How long the final article should be (deviation of 100 except for Chinese).
    language (str): The desired language for the prompt.
    The valid options are "en", "de", "ru", "id", "zh".

    Returns:
    The prompt (str).
    """

    if language == "en":

        prompt = (
            f"Write a news article with the following headline in English: '{title}'. "
            f"Start your article with the following sentence: '{sentence}'. "
            "Do not print the title and do not include separate headlines in the article."
            f"The article should be approximately {article_length} words, with the maximum difference of 100 words."
        )

    elif language == "de":

        prompt = (
            f"Schreiben Sie einen Nachrichtenartikel mit der folgenden Überschrift: '{title}'. "
            f"Beginnen Sie Ihren Artikel mit folgendem Satz: '{sentence}'. "
            "Drucken Sie den Titel nicht und fügen Sie keine separaten Überschriften in den Artikel ein."
            f"Der Artikel sollte ungefähr {article_length} Wörter lang sein, mit einer maximalen Abweichung von 100 Wörtern."
        )

    elif language == "ru":

        prompt = (
            f"Напишите новостную статью со следующим заголовком: '{title}'. "
            f"Начните свою статью со следующего предложения: '{sentence}'. "
            "Не печатайте заголовок и не включайте отдельные подзаголовки в статью."
            f"Статья должна быть приблизительно {article_length} слов, с максимальной разницей в 100 слов."
        )

    elif language == "zh":

        prompt = (
            f"用以下标题写一篇新闻文章: '{title}'。"
            f"用以下句子开始你的文章: '{sentence}'。"
            "不要打印标题，也不要在文章中包含单独的标题。"
            f"文章的长度应大约为 {article_length} 字符，最大差异为 50 字符。"
        )

    elif language == "id":

        prompt = (
            f"Tulislah artikel berita dengan judul berikut: '{title}'. "
            f"Mulailah artikel Anda dengan kalimat berikut: '{sentence}'. "
            "Jangan mencetak judul dan jangan menyertakan judul terpisah di dalam artikel."
            f"Artikel tersebut harus memiliki panjang sekitar {article_length} kata, dengan perbedaan maksimal 100 kata."
        )

    return prompt
