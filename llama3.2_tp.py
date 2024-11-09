from groq import Groq

client = Groq(api_key="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")


completion = client.chat.completions.create(
    model="llama-3.1-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": """Please provide me a json of object classes in that dataset as keys and necessary visual attributes, object parts, sub-objects, sub-classes , alternative classnames or anything that helps you to identify its objects from an image.
Those visual features should be unique to each class and should help to differentiate between classes. There may be classes like background, unlabelled or other areas. You can provide visual feature descriptions for them as well using the context of other classes
Provide 15 visual feature list per class as the values of the dictionary. Use simple Language
Given below are the object class list.
["background of nuclei on a slide", "nuclei in cells"]"""
        },
        {
            "role": "assistant",
            "content": "```json"
        }
    ],
    stop="```",
    
)

# for chunk in completion:
#     print(chunk.choices[0].delta.content or "", end="")


print(completion.choices[0].message.content)

with open("jsons/didula.json", "w") as f:
    f.write(completion.choices[0].message.content)



    
x = """In a game of object detection from images, I'm given an image patch of an object. I'm required to identify the object correctly. Think of visual features that enable me to identify the object from a class list correctly only from the patch.
You are allowed to provide me visual attributes. Most importantly each attributes you give should help to uniquely identify the object class from the image patch. Clearly differentiating classes. So there could not be the same attribute in two different classes.
Since only a patch is given it would be useful if the attributes can describe texture, typical colors etc. Of course you don't have to limit for these and don't provide if it is not useful.
Provide me a json of object classes in that dataset as keys and necessary visual attributes that helps you to identify its objects from an image patch. Provide 5 descriptive attributes list per class as the values of the dictionary. 
There may be patches from background, unlabelled or other areas. You can provide attributes for them as well using the context of other classes
Given below are the object class list."""


y = """Please provide me a json of object classes in that dataset as keys and necessary alternative classnames that helps you to identify its objects from an image.
The alternative classnames should be unique to each class and should help to differentiate between classes. There may be classes like background, unlabelled or other areas. You can provide alternative classnames for them as well using the context of other classes
Provide 5 alternative classnames list per class as the values of the dictionary.  
['unlabled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
"""

z = """Please provide me a json of object classes in that dataset as keys and necessary alternative classnames that helps you to identify its objects from an image.
The alternative classnames should be unique to each class and should help to differentiate between classes. There may be classes like background, unlabelled or other areas. You can provide alternative classnames for them as well using the context of other classes
Provide 5 alternative classnames list per class as the values of the dictionary.  Use simple Language
['unlabled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']"""

