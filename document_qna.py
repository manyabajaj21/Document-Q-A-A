import torch
import gradio as gr
from transformers import pipeline


model_path = "models/roberta_local"


# question_answer = pipeline(
#     "question-answering",
#     model=model_path,
#     tokenizer=model_path,
#     local_files_only=True
# )

question_answer=pipeline("question-answering",model="deepset/roberta-base-squad2")




def read_file_content(file_obj):
    try:
        with open(file_obj.name, 'r', encoding='utf-8') as file:
            context = file.read()
        return context
    except Exception as e:
        return f"An error occurred: {e}"

def get_answer(file, question):
    context = read_file_content(file)
    
    # Check if reading was successful
    if context.startswith("An error occurred"):
        return context
    
    result = question_answer(question=question, context=context)
    return result["answer"]

# Gradio interface
demo = gr.Interface(
    fn=get_answer,
    inputs=[
        gr.File(label="Upload your file"),
        gr.Textbox(label="Input your question", lines=1)
    ],
    outputs=gr.Textbox(label="Answer", lines=1),
    title="@GenAiLearniverse Project: Q and A",
    description="This application will be used to answer questions based on context provided."
)

# Launch the app
demo.launch()
