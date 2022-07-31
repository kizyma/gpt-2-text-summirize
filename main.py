import gradio as gr
from summarizer import TransformerSummarizer

title = "Summarizer"
description = """
This demo is GPT-2 based Summarizer, 
works with English, Ukrainian and Russian (and a few other languages too, it`s GPT-2 after all.
"""


def start_fn(article_input: str) -> str:
    """
    GPT-2 based solution, input full text, output summirized text
    :param article_input:
    :return:
    """
    GPT2_model = TransformerSummarizer(transformer_type="GPT2", transformer_model_key="gpt2-medium")
    full = ''.join(GPT2_model(article_input, min_length=60))
    return full


face = gr.Interface(fn=start_fn,
                    inputs=gr.inputs.Textbox(lines=2, placeholder="Paste article here.", label='Input Article'),
                    outputs=gr.inputs.Textbox(lines=2, placeholder="Summarized article here.", label='Summarized '
                                                                                                     'Article'),
                    title=title,
                    description=description,)
face.launch()
