import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
import random
import os
import time
from dotenv import load_dotenv
import PIL

load_dotenv()


class ImageGeneratorEngine:
    def __init__(
        self,
        number_of_images=1,
        aspect_ratio="1:1",
        language="auto",
        safety_filter_level="block_some",
        person_generation="allow_adult",
        delay=30
    ):
        vertexai.init(
            project=os.environ["GOOGLE_CLOUD_PROJECT"], 
            location=os.environ["GOOGLE_CLOUD_LOCATION"]
        )
        self.model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")
        self.number_of_images = number_of_images
        self.aspect_ratio = aspect_ratio
        self.language = language
        self.safety_filter_level = safety_filter_level
        self.person_generation = person_generation
        self.delay = delay

    def _generate(self, prompt, save_dir, image_name_prefix):
        os.makedirs(save_dir, exist_ok=True)

        images = self.model.generate_images(
            prompt=prompt,
            number_of_images=self.number_of_images,
            language=self.language,
            aspect_ratio=self.aspect_ratio,
            safety_filter_level=self.safety_filter_level,
            person_generation=self.person_generation,
        )

        for idx, image in enumerate(images):
            image_name = f"{image_name_prefix}_{idx}.png"
            image_path = os.path.join(save_dir, image_name)
            try:
                image.save(image_path)
                print(f"Image saved: {image_path}")
            except Exception as e:
                print(f"Error saving {image_path}: {e}")

        return images

    def run(self, prompt:str, save_dir="assets", image_name_prefix="image"):
        try:
            base_dir = f"{save_dir}"
            os.makedirs(base_dir, exist_ok=True)
            images = self._generate(prompt, base_dir, image_name_prefix)
            return images
        except Exception as e:
            return []