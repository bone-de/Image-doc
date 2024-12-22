import os
import base64
import time
import asyncio
import aiohttp
from datetime import datetime
from tqdm import tqdm
from openai import AsyncOpenAI
import logging
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import io

class ImageProcessor:
    def __init__(self, image_dir, output_file, api_key, base_url, max_workers=10):
        self.image_dir = image_dir
        self.output_file = output_file
        self.api_key = api_key
        self.base_url = base_url
        self.max_workers = max_workers
        self.setup_logging()
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('processing.log'),
                logging.StreamHandler()
            ]
        )

    def optimize_image(self, image_path, max_size=800):
        """优化图片大小，减少传输数据量"""
        try:
            with Image.open(image_path) as img:
                # 转换为RGB模式
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 调整图片大小
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # 保存到内存中
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85, optimize=True)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logging.error(f"Error optimizing image {image_path}: {str(e)}")
            return None

    async def process_single_image(self, session, image_name):
        try:
            image_path = os.path.join(self.image_dir, image_name)
            
            # 使用线程池处理图片优化
            with ThreadPoolExecutor() as executor:
                base64_image = await asyncio.get_event_loop().run_in_executor(
                    executor, self.optimize_image, image_path
                )
            
            if not base64_image:
                return None

            response = await self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "识别图片的文字："},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
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

    async def process_batch(self, batch):
        async with aiohttp.ClientSession() as session:
            tasks = [self.process_single_image(session, img) for img in batch]
            return await asyncio.gather(*tasks)

    def chunk_list(self, lst, chunk_size):
        """将列表分成更小的批次"""
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    async def process_images(self):
        start_time = time.time()
        
        try:
            images = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            
            if not images:
                logging.warning(f"No images found in {self.image_dir}")
                return
            
            logging.info(f"Found {len(images)} images to process")
            
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(f"处理开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n")
            
            # 将图片列表分成批次
            batches = self.chunk_list(images, self.max_workers)
            
            with tqdm(total=len(images), desc="Processing images") as pbar:
                for batch in batches:
                    results = await self.process_batch(batch)
                    for result in results:
                        if result:
                            with open(self.output_file, 'a', encoding='utf-8') as f:
                                f.write(result)
                    pbar.update(len(batch))
            
            elapsed_time = time.time() - start_time
            summary = f"\n总处理时间: {elapsed_time:.2f}秒\n处理的图片数量: {len(images)}\n平均每张图片处理时间: {elapsed_time/len(images):.2f}秒\n"
            
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(summary)
            
            logging.info(f"Processing completed. Results saved to {self.output_file}")
            
        except Exception as e:
            logging.error(f"Error during processing: {str(e)}")

async def main():
    processor = ImageProcessor(
        image_dir="images",
        output_file="results.txt",
        api_key="sk-xxxljCFy",
        base_url="https://open.api.gu28.top/v1",
        max_workers=20  # 增加并发数
    )
    await processor.process_images()

if __name__ == "__main__":
    asyncio.run(main())
