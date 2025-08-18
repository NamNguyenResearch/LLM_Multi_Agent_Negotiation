from openai import OpenAI

client = OpenAI(
   base_url = 'http://localhost:11434/v1',
   api_key='ollama', # required, but unused
)

client1 = OpenAI(
   base_url = 'http://localhost:11434/v1',
   api_key='ollama', # required, but unused
)

# response = client.chat.completions.create(
#  model="llama2",
#  messages=[
#    {"role": "system", "content": "You are a helpful assistant."},
#    {"role": "user", "content": "What is the capital of Vietnam? Describe its main attractions."},
#  ]
# )

response1 = client1.chat.completions.create(
 model="llama2",
 messages=[
   {"role": "system", "content": "You are trying to sell a book to me with title 'The Great Gatsby' with higher price as possible. You bought it for $10 and you want to sell it for $100. You are a good salesman. You are trying to convince me to buy it."},
   {"role": "user", "content": "How much for the book?"},
 ]
)
      
print(response1.choices[0].message.content)