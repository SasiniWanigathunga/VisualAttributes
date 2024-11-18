from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# device = "cuda:1" # the device to load the model onto
device = 'cpu'

def generate_prompt(category_name: str, class_list: list, photo_type: str) -> str:
    classes = class_list.copy()
    classes.remove(f'{category_name}')
    print(classes)
    messages = [
        {"role": "user", "content": f"What are useful visual attributes for distinguishing a lemur from {' , '.join(classes)} in a photo?"},
        {"role": "assistant", "content": """There are several useful visual attributes to tell there is a lemur in a photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet"""},
        {"role": "user", "content": f"What are useful visual attributes for distinguishing a television from {' , '.join(classes)} in a photo?"},
        {"role": "assistant", "content": """There are several useful visual attributes to tell there is a television in a photo:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control"""},
        {"role": "user", "content": f"What are useful visual attributes for distinguishing a {category_name} from {' , '.join(classes)} in a {photo_type}?"},
        {"role": "assistant", "content": f"""There are several useful visual attributes to tell there is a {category_name} in a {photo_type}:
-"""}
    ]
    return messages

def string_to_list(result):
    answer = result.split("[/INST]")[-1]
    answer = answer.split("- ")[1:]
    attributes = []
    for a in answer:
        if a == "" or a == " ":
            continue
        if "\n" in a:
            a = a.replace("\n", "")
        if "</s>" in a:
            a = a.replace("</s>", "")
        attributes.append(a)
    return attributes

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

def obtain_descriptors_and_save(filename, class_list, photo_type):
    prompts = [generate_prompt(category.replace('_', ' '), class_list, photo_type) for category in class_list]
    attributes_dict = {}
    for prompt in prompts:
        category_name = class_list[prompts.index(prompt)]
        encodeds = tokenizer.apply_chat_template(prompt, return_tensors="pt")
        model_inputs = encodeds.to(device)
        model.to(device)
        generated_ids = model.generate(model_inputs, max_new_tokens=400, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)[0]
        attributes = string_to_list(decoded)
        attributes_dict[category_name] = attributes
    # save descriptors to json file
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, "w") as file:
        json.dump(attributes_dict, file)

class_list = ['background of night photos taken while driving', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
filename = "jsons/mistral7b.json"
photo_type = "urban city photo taken while driving in night"
obtain_descriptors_and_save(filename, class_list, photo_type)
