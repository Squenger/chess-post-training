"""Hugging Face Space front page: shows the assignment (README.md) and the write-up template.

The Space repository itself is the homework repository: clone it to get the code.
"""

import re
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent


def read_md(name: str) -> str:
    text = (HERE / name).read_text()
    return re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)  # drop the Space YAML header


with gr.Blocks(title="Chess Post-Training Homework") as demo:
    with gr.Tab("Assignment"):
        gr.Markdown(read_md("README.md"))
    with gr.Tab("Write-up template"):
        gr.Markdown("Copy this into `WRITEUP.md` and fill it in.")
        gr.Code(read_md("TEMPLATE_WRITEUP.md"), language="markdown")

if __name__ == "__main__":
    demo.launch()
