from groq import Groq

client = Groq(api_key="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")
completion = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[
        {
            "role": "system",
            "content": "Please follow the below format only. "
        },
        {
            "role": "user",
            "content": "Q: What are useful visual features for distinguishing a lemur in a photo?\nA: There are several useful visual features to tell there is a lemur in a photo:\n- four-limbed primate\n- black, grey, white, brown, or red-brown\n- wet and hairless nose with curved nostrils\n- long tail\n- large eyes\n- furry bodies\n- clawed hands and feet\n\nQ: What are useful visual features for distinguishing a television in a photo?\nA: There are several useful visual features to tell there is a television in a photo:\n- electronic device\n- black or grey\n- a large, rectangular screen\n- a stand or mount to support the screen\n- one or more speakers\n- a power cord\n- input ports for connecting to other devices\n- a remote control\n\nQ: What are useful features for distinguishing a fashion designer in a photo?\nA: There are several useful visual features to tell there is a fashion designer in a photo:\n-"
        }
    ],
    temperature=1,
    max_tokens=1024,
    top_p=1,
    stream=True,
    stop=None,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")