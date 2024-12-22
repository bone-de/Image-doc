import os
import base64
import time
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures
import logging

class ImageProcessor:
    def __init__(self, image_dir, output_file, api_key, base_url):
        self.image_dir = image_dir
        self.output_file = output_file
        self.api_key = api_key
        self.base_url = base_url
        self.setup_logging()
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('processing.log'),
                logging.StreamHandler()
            ]
        )

    def encode_image(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logging.error(f"Error encoding image {image_path}: {str(e)}")
            return None

    def process_single_image(self, image_name):
        try:
            image_path = os.path.join(self.image_dir, image_name)
            base64_image = self.encode_image(image_path)
            
            if not base64_image:
                return None

            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "识别图片的文字："},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
            )
            
            result = response.choices[0].message.content.replace("*", "").replace("#", "").replace(" ", "")
            return f"文件名: {image_name}\n处理结果: {result}\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n"
            
        except Exception as e:
            logging.error(f"Error processing image {image_name}: {str(e)}")
            return None

    def process_images(self):
        start_time = time.time()
        
        try:
            images = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            
            if not images:
                logging.warning(f"No images found in {self.image_dir}")
                return
            
            logging.info(f"Found {len(images)} images to process")
            
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(f"处理开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(self.process_single_image, img): img for img in images}
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(images), desc="Processing images"):
                    result = future.result()
                    if result:
                        with open(self.output_file, 'a', encoding='utf-8') as f:
                            f.write(result)
            
            elapsed_time = time.time() - start_time
            summary = f"\n总处理时间: {elapsed_time:.2f}秒\n处理的图片数量: {len(images)}\n平均每张图片处理时间: {elapsed_time/len(images):.2f}秒\n"
            
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(summary)
            
            logging.info(f"Processing completed. Results saved to {self.output_file}")
            
        except Exception as e:
            logging.error(f"Error during processing: {str(e)}")

def main():
    processor = ImageProcessor(
        image_dir="images",
        output_file="results.txt",
        api_key="sk-xxx",
        base_url="https://open.api.gu28.top/v1"
    )
    processor.process_images()

if __name__ == "__main__":
    main()
