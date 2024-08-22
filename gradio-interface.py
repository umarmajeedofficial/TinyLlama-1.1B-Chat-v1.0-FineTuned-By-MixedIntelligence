import torch
from transformers import pipeline
import gradio as gr

# Initialize the model pipeline
pipe = pipeline("text-generation", model="umarmajeedofficial/TinyLlama-1.1B-Chat-v1.0-FineTuned-By-MixedIntelligence", torch_dtype=torch.bfloat16, device_map="auto")

# Function to generate response
def generate_response(question):
    # Define messages with a clear prompt for a concise answer
    messages = [
        {"role": "system", "content": "You are an expert in emergency situations and environmental issues. Provide concise and direct answers."},
        {"role": "user", "content": question}
    ]

    # Generate the prompt
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate the output
    outputs = pipe(prompt, max_new_tokens=150, do_sample=False)  # Reduced tokens for concise answers

    # Post-process the output to clean up the response
    generated_text = outputs[0]["generated_text"]

    # Clean up the response
    # If the response starts with system prompt or question, strip it out
    start_index = generated_text.find(question)
    if start_index != -1:
        clean_response = generated_text[start_index:].strip()
    else:
        clean_response = generated_text.strip()

    # Optional: Remove any unwanted ending marks like `</s>`
    clean_response = clean_response.replace("</s>", "").strip()

    return clean_response

# Gradio UI
def qa_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Emergency Helper")
        gr.Markdown("Developed by Mixed Intelligence Team")

        with gr.Row():
            with gr.Column():
                emergency_question = gr.Textbox(label="Ask your question about emergency situations or environmental issues", placeholder="e.g., How to survive in an earthquake?")
                submit_btn = gr.Button("Submit")
                output = gr.Textbox(label="Response", placeholder="The answer will appear here...", lines=5)

        submit_btn.click(generate_response, inputs=emergency_question, outputs=output)

    return demo

# Launch the Gradio UI
demo = qa_interface()
demo.launch()
