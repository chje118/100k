OPENSLIDE_PATH = r'D:\openslide-bin-4.0.0.2-windows-x64\bin'
import os
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
import easyocr
import numpy as np
import random
from PIL import Image
import re
from tqdm import tqdm
import pickle

reader = easyocr.Reader(['en']) # load the model into memory (english ('en') and danish ('da'))

class SlideLabelOCR:
    """ Extract rekvnr and stain from WSI label images using EasyOCR. """
    def __init__(self, slide_path):
        self.slide_path = slide_path
        self.slide = openslide.OpenSlide(slide_path)
        self.label_images = self.slide.associated_images
        self.img = self.get_label_image()
        self.results = self.extract_text_from_label()
        
    def get_label_image(self, label_key='label'):
        if label_key not in self.label_images:
            raise ValueError(f"No associated label image found for key: {label_key}")
        return self.label_images[label_key]
        
    def rotate_image(self):
        """Rotate the label by 180 degrees."""
        self.img = self.img.rotate(180, expand=True) 
        
    def show_label_image(self):
        self.img.show()
        
    def extract_text_from_label(self):
        """ Run OCR on current image. """
        img_array = np.array(self.img)
        self.results = reader.readtext(img_array)
        return self.results
    
    def print_results(self):
        for (_, text, conf) in self.results:
            print(f"Detected text: {text} with confidence: {conf:.2f}")

    def _find_rekvnr(self):
        """ Search rekvnr pattern inside OCR results. """
        all_text = " ".join([text for (_, text, _) in self.results])
        match = re.search(r'N.+?(1[23])-(\d{6})', all_text)
        if match:
            return match.group(1) + match.group(2)
        return None

    def get_rekvnr(self):
        """ Try OCR. If no valid rekvnr found → rotate 180° and try again."""
        self.extract_text_from_label()
        rekvnr = self._find_rekvnr()
        if rekvnr:
            return rekvnr

        print(f"Rekvnr not found for {self.slide_path}. Rotating image 180° and retrying...")
        self.rotate_image()
        self.extract_text_from_label()
        rekvnr = self._find_rekvnr()

        if rekvnr:
            print(f"Rekvnr found after rotation: {rekvnr}")
            return rekvnr

        print("Rekvnr still not found after rotation.")
        return None
    
    def get_stain(self):
        """ Extract stain as the 2nd last recognized line """
        if len(self.results) < 2:
            print(f"Not enough OCR lines to extract stain for {self.slide_path}")
            return None
        _, stain_text, _ = self.results[-2]
        return stain_text


class LabelOCRCache: 
    def __init__(self, wsi_paths: list[str], cache_file: str):
        self.wsi_paths = wsi_paths
        self.cache_file = cache_file
        self.ocr_results = {} # { filename: {"rekvnr":..., "stain":...} }
        self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                self.ocr_results = pickle.load(f)
                print(f"Loaded {len(self.ocr_results)} previously processed entries.")
        else:
            print("No cache found.")

    def save_cache(self):
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.ocr_results, f)
        print(f"Saved cache with {len(self.ocr_results)} entries.")

    def extract_with_ocr(self, path):
        try:
            ocr = SlideLabelOCR(path)
            rekvnr = ocr.get_rekvnr()
            stain = ocr.get_stain()
            print(f"→ OCR for {os.path.basename(path)}: rekvnr={rekvnr}, stain={stain}")
            return rekvnr, stain
        except Exception as e:
            print(f"ERROR processing {path}: {e}")
            return None, None

    def run(self):
        remaining = [p for p in self.wsi_paths if p not in self.ocr_results]
        print(f"Already cached: {len(self.ocr_results)}")
        print(f"Remaining to process: {len(remaining)}")

        for i, path in enumerate(tqdm(remaining, desc="OCR progress")):
            rekvnr, stain = self.extract_with_ocr(path)
            self.ocr_results[path] = {"rekvnr": rekvnr, "stain": stain}

            # Save every 10 slides
            if i % 10 == 0:
                self.save_cache()

        # Final save
        self.save_cache()

        # Convert dict → DataFrame
        df = pd.DataFrame.from_dict(self.ocr_results, orient="index")
        df.index.name = "filename"
        return df

    
if __name__ == "__main__":
    # Example usage:
    slide_dir = r"\\regsj.intern\appl\Deep_Visual_Proteomics\Slides 15.01.2024"
    slide_paths = [os.path.join(slide_dir, f) for f in os.listdir(slide_dir) if f.endswith('.mrxs')]
    print(f"Number of available slides in directory: {len(slide_paths)}")
    
    # Single Slide OCR
    slide_path = random.choice(slide_paths)
    print(f"Trying to open slide: {slide_path}")
    
    ocr = SlideLabelOCR(slide_path)
    ocr.show_label_image('label')
    ocr.print_results()
    print(f"Extracted rekvnr: {ocr.get_rekvnr()}")   

    # OCR cache
    CACHE_FILE = "wsi_ocr_cache.pkl"

    cache = LabelOCRCache(slide_paths, CACHE_FILE)
    df_ocr = cache.run()
    print(df_ocr.head())