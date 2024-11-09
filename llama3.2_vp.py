from groq import Groq
import os
from PIL import Image
import base64

client = Groq(api_key="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")

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = '/home/ulindu/datasets/Corrosion Condition State Classification/original/Test/images/0.jpeg'
base64_image = encode_image(image_path)

def generate_prompt(class_list):
    return f"""
    Provide a JSON dictionary of 5 visual attributes to distinguish each of the following classes in a segmentation task. 
    Attributes can include shape, orientation, primary physical characteristics, material, texture, or typical colors. 
    Ensure that the attributes for each class are distinct from one another.
    
    Class list: {class_list}
    Example output format:
    {{
        "class_name_1": ["attribute1", "attribute2", "attribute3", "attribute4", "attribute5"],
        "class_name_2": ["attribute1", "attribute2", "attribute3", "attribute4", "attribute5"]
    }}
    """

def obtain_descriptors_and_save(filename, class_list):
    prompt = generate_prompt(class_list)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    }
                ]
            }
        ],
        model="llama-3.2-90b-vision-preview",
        response_format={"type": "json_object"},
    )

    result = chat_completion.choices[0].message.content
    print(result)

    with open(filename, "w") as f:
        f.write(result)

# Running the function with the class list
obtain_descriptors_and_save("jsons/corrosion_cs_sem_seg_test.json", [
    'others (regions such as the concrete surfaces, metal surfaces or environment)',
    'steel with fair corrosion',
    'steel with poor corrosion',
    'steel with severe corrosion'
])
