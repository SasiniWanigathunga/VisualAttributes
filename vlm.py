from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
import sys
import os

def generate_prompt(category_name: str, class_list: list) -> str:
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

class Prompting:
    def __init__(self):
        self.model = 'OpenGVLab/InternVL2-2B'
        self.pipe = pipeline(self.model, backend_config=TurbomindEngineConfig(session_len=8192), )  

    
    def get_prompt(self, prompt, img_path: list):
        print(img_path)
        image_urls=img_path
        prompts = [(prompt, load_image(img_url)) for img_url in image_urls] 
        print(prompts)     
        
        print(self.pipe(prompts))
        response = self.pipe(prompts)[0].text            
        response = self.stringtolist(response)            
        return response


    def stringtolist(self, description):
        output_list = []
        description = description.split('\n') # type: ignore
        classes = 0
        for descriptor in description:
            if descriptor.startswith('Q:'):
                classes += 1
                if classes > 1:
                    break
            while descriptor.startswith(' '):
                descriptor = descriptor[1:]
            if descriptor != '' and type(descriptor) == str and not descriptor.startswith('Q:') and not descriptor.startswith('A:'):
                while descriptor.startswith(' ') or descriptor.startswith('-') or descriptor.startswith('*'):
                    descriptor = descriptor[1:]
                if descriptor != '':
                    output_list.append(descriptor)
        return output_list
    
def main():
    class_list = ['Black_footed_Albatross']
    print(os.getcwd())

    img_paths = ['/home/ulindu/datasets/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg']
    vlm_generate = Prompting()
    
    prompts = [generate_prompt(category.replace('_', ' '), class_list) for category in class_list]

    for i in range(len(class_list)):
        response = vlm_generate.get_prompt(prompts[i], img_paths)
        print(response)

if __name__ == '__main__':
    main()