import openai

system_content = "You are an AI model specializing in environmental issues. Your task is to generate question-answer pairs on the environmental issue named 'Air Pollution'. Generate up to 60 question-answer pairs"
user_content = "Generate 60 question-answer pairs about Air Pollution."

client = openai.OpenAI(
    api_key="AI/ML API",
    base_url="https://api.aimlapi.com",
)

chat_completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ],
    temperature=0.7,
    max_tokens=3000,
)

response = chat_completion.choices[0].message.content
print("AI/ML API:\n", response)

