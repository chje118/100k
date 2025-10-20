# Use easyOCR to extract rekvnr from slide labels
import easyocr
import openslide
import numpy as np
import os
import random
from PIL import Image
import re

reader = easyocr.Reader(['en', 'da']) # load the model into memory (english and danish)

class SlideLabelOCR:
    def __init__(self, slide_path):
        # TODO: Rely on wsireader (works natively with LazySlide) instead of openslide
        self.slide = openslide.OpenSlide(slide_path)
        self.label_images = self.slide.associated_images
        self.results = self.extract_text_from_label()

    def get_label_image(self, label_key='label'):
        if label_key in self.label_images:
            label_image = self.label_images[label_key]
            label_image = label_image.rotate(180, expand=True)
            return label_image
        else:
            raise ValueError(f"No associated image found for key: {label_key}")
        
    def show_label_image(self, label_key='label'):
        label_image = self.get_label_image(label_key)
        label_image.show()

    def extract_text_from_label(self, label_key='label'):
        label_image = self.get_label_image(label_key)
        # Convert PIL image to numpy array
        label_array = np.array(label_image)
        # Use easyOCR to read text from the image
        result = reader.readtext(label_array)
        return result
    
    def print_results(self):
        for (_, text, conf) in self.results:
            print(f"Detected text: {text} with confidence: {conf}")

    def get_rekvnr(self):
        all_text = " ".join([text for (_, text, _) in self.results])
        match = re.search(r'N.+?(1[23])-(\d{6})', all_text)
        if match:
            eight_digits = match.group(1) + match.group(2)
            return eight_digits
        else:
            print("Rekvnr pattern not found.")
            return None
     
# Example usage:
if __name__ == "__main__":
    slide_dir = r"\\regsj.intern\appl\Deep_Visual_Proteomics\Slides 15.01.2024"
    slide_paths = [os.path.join(slide_dir, f) for f in os.listdir(slide_dir) if f.endswith('.mrxs')]
    print(f"Number of available slides in directory: {len(slide_paths)}")
    slide_path = random.choice(slide_paths)
    print(f"Trying to open slide: {slide_path}")

    ocr = SlideLabelOCR(slide_path)
    ocr.show_label_image('label')
    ocr.print_results()
    print(f"Extracted rekvnr: {ocr.get_rekvnr()}")