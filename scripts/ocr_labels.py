# Use easyOCR to extract rekvnr from slide labels
import easyocr
import openslide
import numpy as np

reader = easyocr.Reader(['en', 'da']) # load the model into memory (english and danish)

class SlideLabelOCR:
    def __init__(self, slide_path):
        self.slide = openslide.OpenSlide(slide_path)
        self.label_images = self.slide.associated_images
        self.results = self.extract_text_from_label()

    def get_label_image(self, label_key='label'):
        if label_key in self.label_images:
            return self.label_images[label_key]
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
    
# Example usage:
if __name__ == "__main__":
    slide_path = 'C:/Users/chris/OneDrive/Dokumenter/SDU/Master\'s Thesis Project/Sample MRXS/Mirax2.2-3.mrxs'
    ocr = SlideLabelOCR(slide_path)
    ocr.show_label_image('label')
    ocr.print_results()