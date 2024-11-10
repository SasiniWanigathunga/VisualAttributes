from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

def generate_prompt(category_name: str, class_list: list, photo_type: str) -> str:
    classes = class_list.copy()
    classes.remove(f'{category_name}')
    print(classes)
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are useful visual attributes for distinguishing a lemur from {' , '.join(classes)} in a photo?
A: There are several useful visual attributes to tell there is a lemur in a photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual attributes for distinguishing a television from {' , '.join(classes)} in a photo?
A: There are several useful visual attributes to tell there is a television in a photo:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control

Q: What are useful visual attributes for distinguishing a {category_name} from {' , '.join(classes)} in a {photo_type}?
A: There are several useful visual attributes to tell there is a {category_name} in a {photo_type}:
-
"""

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    torch_dtype=torch.bfloat16
)

# input_text = "Write me a poem about Machine Learning."
# input_ids = tokenizer(input_text, return_tensors="pt")

# outputs = model.generate(**input_ids)
# print(tokenizer.decode(outputs[0]))

def string_to_list(result):
    answer = result.split("Q:")[-1]
    answer = answer.split("- ")[1:]
    attributes = []
    for a in answer:
        if a == "" or a == " ":
            continue
        elif "\n" in a:
            a = a.replace("\n", "")
            attributes.append(a)
        elif "<eos>" in a:
            a = a.replace("<eos>", "")
            attributes.append(a)
        else:
            attributes.append(a)
    print(attributes)
    return attributes

def obtain_descriptors_and_save(filename, class_list, photo_type):
    prompts = [generate_prompt(category.replace('_', ' '), class_list, photo_type) for category in class_list]
    attributes_dict = {}
    for prompt in prompts:
        category_name = class_list[prompts.index(prompt)]
        input_ids = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**input_ids, max_new_tokens=400)
        result = tokenizer.decode(outputs[0])
        with open("gemma2b_prompts.txt", "a") as file:
            print(result, file=file)
        attributes = string_to_list(result)
        attributes_dict[category_name] = attributes
    # save descriptors to json file
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, "w") as file:
        json.dump(attributes_dict, file)

class_list = ['background of night photos taken while driving', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
filename = "jsons/gemma2b.json"
photo_type = "urban city photo taken while driving in night"
obtain_descriptors_and_save(filename, class_list, photo_type)