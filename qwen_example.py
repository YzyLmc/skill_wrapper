from openai import OpenAI
import os
import base64


client = OpenAI(
    # If the environment variable is not configured, replace the following line with: api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

with open("qwen_example_east_coast_locus_map.png", "rb") as f:
    image_data = f.read()

completion = client.chat.completions.create(
    model="qwen3-vl-235b-a22b-instruct",
    messages=[
        {
            "role":"user",
            "content":[
                {
                    "type":"image_url",
                    "image_url":{
                        "url": "data:image/png;base64," + base64.b64encode(image_data).decode('utf-8')
                    }
                },
                {
                    "type":"text",
                    "text":"What is in this picture?"
                }
            ]
        }
    ],
    stream=True,
    # extra_body={
    #     'enable_thinking': enable_thinking,
    #     "thinking_budget": 81920
    # },
)
breakpoint()
full_content = ""
print("Streaming output content:")
for chunk in completion:
    if chunk.choices[0].delta.content is None:
        continue
    full_content += chunk.choices[0].delta.content
    print(chunk.choices[0].delta.content)
print(f"Complete content: {full_content}")

