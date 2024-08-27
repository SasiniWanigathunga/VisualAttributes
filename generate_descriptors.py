import os
import openai
import json

import itertools

from descriptor_strings import stringtolist

from groq import Groq

client = Groq(api_key="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")

def generate_prompt(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are useful visual attributes for distinguishing a lemur in a photo?
A: There are several useful visual attributes to tell there is a lemur in a photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual attributes for distinguishing a television in a photo?
A: There are several useful visual attributes to tell there is a television in a photo:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control

Q: What are useful visual attributes for distinguishing a {category_name} in a photo?
A: There are several useful visual attributes to tell there is a {category_name} in a photo:
-
"""

# generator 
def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))
        
def obtain_descriptors_and_save(filename, class_list):
    responses = {}
    descriptors = {}
    response_texts = []
    
    prompts = [generate_prompt(category.replace('_', ' ')) for category in class_list]

    for prompt in prompts:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "Please follow the below format only. "
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        answer = ""
        for chunk in completion:
            answer += chunk.choices[0].delta.content or "" + "\n"

        response_texts.append(answer)

        print(response_texts)
    
    descriptors_list = [stringtolist(response_text) for response_text in response_texts]
    descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    # save descriptors to json file
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as fp:
        json.dump(descriptors, fp)
    

# obtain_descriptors_and_save('jsons/ade20k_150_llama70B.json', ["wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed ", "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television receiver", "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag"])
# obtain_descriptors_and_save('example', ["cat", "dog", "person", "chair"])
# obtain_descriptors_and_save('jsons/pascalvoc_20_llama70B.json', ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"])
obtain_descriptors_and_save('jsons/mess_llama70B.json', ['background', 'Black-footed Albatross', 'Laysan Albatross', 'Sooty Albatross', 'Groove billed Ani', 'Crested Auklet', 'Least Auklet', 'Parakeet Auklet', 'Rhinoceros Auklet', 'Brewer Blackbird', 'Red winged Blackbird', 'Rusty Blackbird', 'Yellow-headed Blackbird', 'Bobolink', 'Indigo Bunting', 'Lazuli Bunting', 'Painted Bunting', 'Cardinal', 'Spotted Catbird', 'Gray Catbird', 'Yellow-breasted Chat', 'Eastern Towhee', 'Chuck will Widow', 'Brandt Cormorant', 'Red-faced Cormorant', 'Pelagic Cormorant', 'Bronzed Cowbird', 'Shiny Cowbird', 'Brown Creeper', 'American Crow', 'Fish Crow', 'Black-billed Cuckoo', 'Mangrove Cuckoo', 'Yellow-billed Cuckoo', 'Gray-crowned Rosy-Finch', 'Purple Finch', 'Northern Flicker', 'Acadian Flycatcher', 'Great Crested Flycatcher', 'Least Flycatcher', 'Olive-sided Flycatcher', 'Scissor-tailed Flycatcher', 'Vermilion Flycatcher', 'Yellow-bellied Flycatcher', 'Frigatebird', 'Northern Fulmar', 'Gadwall', 'American Goldfinch', 'European Goldfinch', 'Boat-tailed Grackle', 'Eared Grebe', 'Horned Grebe', 'Pied-billed Grebe', 'Western Grebe', 'Blue Grosbeak', 'Evening Grosbeak', 'Pine Grosbeak', 'Rose-breasted Grosbeak', 'Pigeon Guillemot', 'California Gull', 'Glaucous-winged Gull', 'Heermann Gull', 'Herring Gull', 'Ivory Gull', 'Ring-billed Gull', 'Slaty-backed Gull', 'Western Gull', 'Anna Hummingbird', 'Ruby-throated Hummingbird', 'Rufous Hummingbird', 'Green Violetear', 'Long-tailed Jaeger', 'Pomarine Jaeger', 'Blue Jay', 'Florida Jay', 'Green Jay', 'Dark-eyed Junco', 'Tropical Kingbird', 'Gray Kingbird', 'Belted Kingfisher', 'Green Kingfisher', 'Pied Kingfisher', 'Ringed Kingfisher', 'White-breasted Kingfisher', 'Red-legged Kittiwake', 'Horned Lark', 'Pacific Loon', 'Mallard', 'Western Meadowlark', 'Hooded Merganser', 'Red-breasted Merganser', 'Mockingbird', 'Nighthawk', 'Clark Nutcracker', 'White-breasted Nuthatch', 'Baltimore Oriole', 'Hooded Oriole', 'Orchard Oriole', 'Scott Oriole', 'Ovenbird', 'Brown Pelican', 'White Pelican', 'Western Wood-Pewee', 'Sayornis', 'American Pipit', 'Whip-poor-Will', 'Horned Puffin', 'Common Raven', 'White-necked Raven', 'American Redstart', 'Geococcyx', 'Loggerhead Shrike', 'Great Grey Shrike', 'Baird Sparrow', 'Black-throated Sparrow', 'Brewer Sparrow', 'Chipping Sparrow', 'Clay-colored Sparrow', 'House Sparrow', 'Field Sparrow', 'Fox Sparrow', 'Grasshopper Sparrow', 'Harris Sparrow', 'Henslow Sparrow', 'Le Conte Sparrow', 'Lincoln Sparrow', 'Nelson Sharp-tailed Sparrow', 'Savannah Sparrow', 'Seaside Sparrow', 'Song Sparrow', 'Tree Sparrow', 'Vesper Sparrow', 'White-crowned Sparrow', 'White-throated Sparrow', 'Cape Glossy Starling', 'Bank Swallow', 'Barn Swallow', 'Cliff Swallow', 'Tree Swallow', 'Scarlet Tanager', 'Summer Tanager', 'Artic Tern', 'Black Tern', 'Caspian Tern', 'Common Tern', 'Elegant Tern', 'Forsters Tern', 'Least Tern', 'Green-tailed Towhee', 'Brown Thrasher', 'Sage Thrasher', 'Black-capped Vireo', 'Blue-headed Vireo', 'Philadelphia Vireo', 'Red-eyed Vireo', 'Warbling Vireo', 'White-eyed Vireo', 'Yellow-throated Vireo', 'Bay-breasted Warbler', 'Black-and-white Warbler', 'Black-throated Blue Warbler', 'Blue-winged Warbler', 'Canada Warbler', 'Cape May Warbler', 'Cerulean Warbler', 'Chestnut-sided Warbler', 'Golden-winged Warbler', 'Hooded Warbler', 'Kentucky Warbler', 'Magnolia Warbler', 'Mourning Warbler', 'Myrtle Warbler', 'Nashville Warbler', 'Orange-crowned Warbler', 'Palm Warbler', 'Pine Warbler', 'Prairie Warbler', 'Prothonotary Warbler', 'Swainson Warbler', 'Tennessee Warbler', 'Wilson Warbler', 'Worm-eating Warbler', 'Yellow Warbler', 'Northern Waterthrush', 'Louisiana Waterthrush', 'Bohemian Waxwing', 'Cedar Waxwing', 'American Three-toed Woodpecker', 'Pileated Woodpecker', 'Red-bellied Woodpecker', 'Red-cockaded Woodpecker', 'Red-headed Woodpecker', 'Downy Woodpecker', 'Bewick Wren', 'Cactus Wren', 'Carolina Wren', 'House Wren', 'Marsh Wren', 'Rock Wren', 'Winter Wren', 'Common Yellowthroat'])
