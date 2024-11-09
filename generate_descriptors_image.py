import os
import json

import itertools

from descriptor_strings import stringtolist

from groq import Groq
import base64
import ast  

client = Groq(api_key="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")

# def generate_prompt(category_name: str, class_list: list) -> str:
#     classes = class_list.copy()
#     classes.remove(f'{category_name}')
#     print(classes)
#     # you can replace the examples with whatever you want; these were random and worked, could be improved
#     return f"""Q: What are useful visual attributes for distinguishing a lemur from {' , '.join(classes)} in a photo?
# A: There are several useful visual attributes to tell there is a lemur in a photo:
# - four-limbed primate
# - black, grey, white, brown, or red-brown
# - wet and hairless nose with curved nostrils
# - long tail
# - large eyes
# - furry bodies
# - clawed hands and feet

# Q: What are useful visual attributes for distinguishing a television from {' , '.join(classes)} in a photo?
# A: There are several useful visual attributes to tell there is a television in a photo:
# - electronic device
# - black or grey
# - a large, rectangular screen
# - a stand or mount to support the screen
# - one or more speakers
# - a power cord
# - input ports for connecting to other devices
# - a remote control

# Q: What are useful visual attributes for distinguishing a {category_name} from{' , '.join(classes)} in a photo?
# A: There are several useful visual attributes to tell there is a {category_name} in a photo:
# -
# """

# def generate_prompt(category_name: str, class_list: list) -> str:
#     classes = class_list.copy()
#     print(classes)
#     # you can replace the examples with whatever you want; these were random and worked, could be improved
#     return f"""Provide python dictionary of visual attributes for the given classes to distinguish them from each other when patches of this image are provided: ",".join({classes})

# The keys of the dictionary should be the original classname and as values please provide a list containing 5 visual attributes. Give the dictionary only, no other text.
# """

def generate_prompt(category_name: str, class_list: list) -> str:
    classes = class_list.copy()
    print(classes)
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Give 5 different visual attributes to distinguish the following classes from each other in a segmentation task. Attributes can be shape, orientation, and primary physical characteristics, material, texture, or typical colors. Attributes for one class must differ from other class.
[', '.join({classes})]
Provide a python dictionary where the keys should be the original classname and values should be visual attributes. Give the dictionary only, no other text. 
"""

# generator 
def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
        
def obtain_descriptors_and_save(filename, class_list):    
    prompts = [generate_prompt(category.replace('_', ' '), class_list) for category in class_list]

    image_path = "/home/ulindu/datasets/Corrosion Condition State Classification/original/Test/images/0.jpeg"
    # Getting the base64 string
    base64_image = encode_image(image_path)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts[0]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="llama-3.2-90b-vision-preview",
    )

    response = chat_completion.choices[0].message.content
    print(response)

    # answer = ""
    # for chunk in chat_completion:
    #     answer += chunk.choices[0].delta.content or "" + "\n"

    # response_texts.append(answer)

    # print(response_texts)

    # descriptors_list = [stringtolist(response_text) for response_text in response_texts]
    # descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    # # save descriptors to json file
    # if not filename.endswith('.json'):
    #     filename += '.json'
    # with open(filename, 'w') as fp:
    #     json.dump(descriptors, fp)

    # save responses to json file
    with open(filename, 'w') as json_file:
        json.dump(ast.literal_eval(response), json_file, indent=4)
    

# obtain_descriptors_and_save('jsons/ade150_llama70B__allclasses.json', ["wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed ", "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television receiver", "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag"])
# obtain_descriptors_and_save('jsons/nuclei4', ["background of nuclei on a slide", "nuclei in cells"])
# obtain_descriptors_and_save('jsons/pascalvoc_20_llama70B_allclasses.json', ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"])
# obtain_descriptors_and_save('jsons/mess_llama70B.json', ['background', 'Black-footed Albatross', 'Laysan Albatross', 'Sooty Albatross', 'Groove billed Ani', 'Crested Auklet', 'Least Auklet', 'Parakeet Auklet', 'Rhinoceros Auklet', 'Brewer Blackbird', 'Red winged Blackbird', 'Rusty Blackbird', 'Yellow-headed Blackbird', 'Bobolink', 'Indigo Bunting', 'Lazuli Bunting', 'Painted Bunting', 'Cardinal', 'Spotted Catbird', 'Gray Catbird', 'Yellow-breasted Chat', 'Eastern Towhee', 'Chuck will Widow', 'Brandt Cormorant', 'Red-faced Cormorant', 'Pelagic Cormorant', 'Bronzed Cowbird', 'Shiny Cowbird', 'Brown Creeper', 'American Crow', 'Fish Crow', 'Black-billed Cuckoo', 'Mangrove Cuckoo', 'Yellow-billed Cuckoo', 'Gray-crowned Rosy-Finch', 'Purple Finch', 'Northern Flicker', 'Acadian Flycatcher', 'Great Crested Flycatcher', 'Least Flycatcher', 'Olive-sided Flycatcher', 'Scissor-tailed Flycatcher', 'Vermilion Flycatcher', 'Yellow-bellied Flycatcher', 'Frigatebird', 'Northern Fulmar', 'Gadwall', 'American Goldfinch', 'European Goldfinch', 'Boat-tailed Grackle', 'Eared Grebe', 'Horned Grebe', 'Pied-billed Grebe', 'Western Grebe', 'Blue Grosbeak', 'Evening Grosbeak', 'Pine Grosbeak', 'Rose-breasted Grosbeak', 'Pigeon Guillemot', 'California Gull', 'Glaucous-winged Gull', 'Heermann Gull', 'Herring Gull', 'Ivory Gull', 'Ring-billed Gull', 'Slaty-backed Gull', 'Western Gull', 'Anna Hummingbird', 'Ruby-throated Hummingbird', 'Rufous Hummingbird', 'Green Violetear', 'Long-tailed Jaeger', 'Pomarine Jaeger', 'Blue Jay', 'Florida Jay', 'Green Jay', 'Dark-eyed Junco', 'Tropical Kingbird', 'Gray Kingbird', 'Belted Kingfisher', 'Green Kingfisher', 'Pied Kingfisher', 'Ringed Kingfisher', 'White-breasted Kingfisher', 'Red-legged Kittiwake', 'Horned Lark', 'Pacific Loon', 'Mallard', 'Western Meadowlark', 'Hooded Merganser', 'Red-breasted Merganser', 'Mockingbird', 'Nighthawk', 'Clark Nutcracker', 'White-breasted Nuthatch', 'Baltimore Oriole', 'Hooded Oriole', 'Orchard Oriole', 'Scott Oriole', 'Ovenbird', 'Brown Pelican', 'White Pelican', 'Western Wood-Pewee', 'Sayornis', 'American Pipit', 'Whip-poor-Will', 'Horned Puffin', 'Common Raven', 'White-necked Raven', 'American Redstart', 'Geococcyx', 'Loggerhead Shrike', 'Great Grey Shrike', 'Baird Sparrow', 'Black-throated Sparrow', 'Brewer Sparrow', 'Chipping Sparrow', 'Clay-colored Sparrow', 'House Sparrow', 'Field Sparrow', 'Fox Sparrow', 'Grasshopper Sparrow', 'Harris Sparrow', 'Henslow Sparrow', 'Le Conte Sparrow', 'Lincoln Sparrow', 'Nelson Sharp-tailed Sparrow', 'Savannah Sparrow', 'Seaside Sparrow', 'Song Sparrow', 'Tree Sparrow', 'Vesper Sparrow', 'White-crowned Sparrow', 'White-throated Sparrow', 'Cape Glossy Starling', 'Bank Swallow', 'Barn Swallow', 'Cliff Swallow', 'Tree Swallow', 'Scarlet Tanager', 'Summer Tanager', 'Artic Tern', 'Black Tern', 'Caspian Tern', 'Common Tern', 'Elegant Tern', 'Forsters Tern', 'Least Tern', 'Green-tailed Towhee', 'Brown Thrasher', 'Sage Thrasher', 'Black-capped Vireo', 'Blue-headed Vireo', 'Philadelphia Vireo', 'Red-eyed Vireo', 'Warbling Vireo', 'White-eyed Vireo', 'Yellow-throated Vireo', 'Bay-breasted Warbler', 'Black-and-white Warbler', 'Black-throated Blue Warbler', 'Blue-winged Warbler', 'Canada Warbler', 'Cape May Warbler', 'Cerulean Warbler', 'Chestnut-sided Warbler', 'Golden-winged Warbler', 'Hooded Warbler', 'Kentucky Warbler', 'Magnolia Warbler', 'Mourning Warbler', 'Myrtle Warbler', 'Nashville Warbler', 'Orange-crowned Warbler', 'Palm Warbler', 'Pine Warbler', 'Prairie Warbler', 'Prothonotary Warbler', 'Swainson Warbler', 'Tennessee Warbler', 'Wilson Warbler', 'Worm-eating Warbler', 'Yellow Warbler', 'Northern Waterthrush', 'Louisiana Waterthrush', 'Bohemian Waxwing', 'Cedar Waxwing', 'American Three-toed Woodpecker', 'Pileated Woodpecker', 'Red-bellied Woodpecker', 'Red-cockaded Woodpecker', 'Red-headed Woodpecker', 'Downy Woodpecker', 'Bewick Wren', 'Cactus Wren', 'Carolina Wren', 'House Wren', 'Marsh Wren', 'Rock Wren', 'Winter Wren', 'Common Yellowthroat'])
# obtain_descriptors_and_save('jsons/custom_classes.json', ['background or trash', 'rigid plastic', 'cardboard', 'metal', 'soft plastic'])
# obtain_descriptors_and_save("jsons/kavindu.json", ["background", "candy", "egg tart", "french fries", "chocolate", "biscuit", "popcorn", "pudding", "ice cream", "cheese butter", "cake", "wine", "milkshake", "coffee", "juice", "milk", "tea", "almond", "red beans", "cashew", "dried cranberries", "soy", "walnut", "peanut", "egg", "apple", "date", "apricot", "avocado", "banana", "strawberry", "cherry", "blueberry", "raspberry", "mango", "olives", "peach", "lemon", "pear", "fig", "pineapple", "grape", "kiwi", "melon", "orange", "watermelon", "steak", "pork", "chicken duck", "sausage", "fried meat", "lamb", "sauce", "crab", "fish", "shellfish", "shrimp", "soup", "bread", "corn", "hamburg", "pizza", "hanamaki baozi", "wonton dumplings", "pasta", "noodles", "rice", "pie", "tofu", "eggplant", "potato", "garlic", "cauliflower", "tomato", "kelp", "seaweed", "spring onion", "rape", "ginger", "okra", "lettuce", "pumpkin", "cucumber", "white radish", "carrot", "asparagus", "bamboo shoots", "broccoli", "celery stick", "cilantro mint", "snow peas", "cabbage", "bean sprouts", "onion", "pepper", "green beans", "French beans", "king oyster mushroom", "shiitake", "enoki mushroom", "oyster mushroom", "white button mushroom", "salad", "other ingredients"])
# obtain_descriptors_and_save("jsons/CLASSES.json", ['unlabeled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'])
# obtain_descriptors_and_save("jsons/CLASSES2.json", ['background', 'fire extinguisher', 'backpack', 'drill', 'human'])
# obtain_descriptors_and_save("jsons/CLASSES_OFFICAL.json", ['background', 'fire extinguisher', 'backpack', 'drill', 'rescue randy'])
# obtain_descriptors_and_save("jsons/CLASSES3.json", ['background', 'building-flooded', 'building-non-flooded', 'road-flooded', 'road-non-flooded', 'water', 'tree', 'vehicle', 'pool', 'grass'])

# obtain_descriptors_and_save("jsons/chase_db1_sem_seg_test.json", ['non-blood vessel regions of the retinal image', 'blood vessels'])
# obtain_descriptors_and_save("jsons/dark_zurich_sem_seg_val.json", ['background of night photos taken while driving', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'])
# obtain_descriptors_and_save("jsons/mhp_v1_sem_seg_test.json", ['everything other than body parts or clothes items', 'hat', 'hair', 'sunglasses', 'upper clothes', 'skirt', 'pants', 'dress', 'belt', 'left shoe', 'right shoe', 'face', 'left leg', 'right leg', 'left arm', 'right arm', 'bag', 'scarf', 'torso skin'])
# obtain_descriptors_and_save("jsons/corrosion_cs_sem_seg_test.json", ['regions such as the concrete surfaces, metal surfaces or environment', 'steel with fair corrosion', 'steel with poor corrosion', 'steel with severe corrosion'])
# obtain_descriptors_and_save("jsons/dram_sem_seg_test.json", ['bird', 'boat', 'bottle', 'cat', 'chair', 'cow', 'dog', 'horse', 'person', 'potted-plant', 'sheep', 'background'])
# obtain_descriptors_and_save("jsons/isaid_sem_seg_val.json", ['background of aerial images', 'boat', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court', 'running track field', 'bridge', 'truck or bus', 'car', 'helicopter', 'swimming pool', 'traffic circle', 'soccer field', 'air plane', 'pier'])
# obtain_descriptors_and_save("jsons/fork.json", ["fork"])
# obtain_descriptors_and_save("jsons/cwfid_sem_seg_test.json", ["ground", "crop seedling", "weed"])
# obtain_descriptors_and_save("jsons/kvasir_instrument_sem_seg_test.json", ["gastrointestinal (GI) tract tissue", "endoscopic grasping tool"])
# obtain_descriptors_and_save("jsons/cryonuseg_sem_seg_test.json", ["others", "nuclei in cells"])
# obtain_descriptors_and_save("jsons/coco_attributes.json", ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "banner", "blanket", "branch", "bridge", "building-other", "bush", "cabinet", "cage", "cardboard", "carpet", "ceiling-other", "ceiling-tile", "cloth", "clothes", "clouds", "counter", "cupboard", "curtain", "desk-stuff", "dirt", "door-stuff", "fence", "floor-marble", "floor-other", "floor-stone", "floor-tile", "floor-wood", "flower", "fog", "food-other", "fruit", "furniture-other", "grass", "gravel", "ground-other", "hill", "house", "leaves", "light", "mat", "metal", "mirror-stuff", "moss", "mountain", "mud", "napkin", "net", "paper", "pavement", "pillow", "plant-other", "plastic", "platform", "playingfield", "railing", "railroad", "river", "road", "rock", "roof", "rug", "salad", "sand", "sea", "shelf", "sky-other", "skyscraper", "snow", "solid-other", "stairs", "stone", "straw", "structural-other", "table", "tent", "textile-other", "towel", "tree", "vegetable", "wall-brick", "wall-concrete", "wall-other", "wall-panel", "wall-stone", "wall-tile", "wall-wood", "water-other", "waterdrops", "window-blind", "window-other", "wood"])


obtain_descriptors_and_save("jsons/corrosion_cs_sem_seg_test.json", ['regions such as the concrete surfaces, metal surfaces or environment', 'steel with fair corrosion', 'steel with poor corrosion', 'steel with severe corrosion'])
