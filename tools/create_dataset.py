from data import load_dataset
from PIL import Image
import io
import random
Image.MAX_IMAGE_PIXELS = None



def resize_and_convert_to_webp(image, max_size=512):
    img = image.convert('RGB')
    width, height = img.size
    if width > height:
        if width > max_size:
            height = int(height * (max_size / width))
            width = max_size
    else:
        if height > max_size:
            width = int(width * (max_size / height))
            height = max_size
    resampling_methods = [
        Image.NEAREST,
        Image.BOX,
        Image.BILINEAR,
        Image.HAMMING,
        Image.BICUBIC,
        Image.LANCZOS
    ]
    chosen_method = random.choice(resampling_methods)
    img = img.resize((width, height), chosen_method)
    buffer = io.BytesIO()
    quality = 90 #random.randint(50, 100)
    img.save(buffer, format="WebP", quality=quality)
    return buffer.getvalue()

def process_cc12m():
    raw_dataset = load_dataset('./cc12m')['train'] # laion/conceptual-captions-12m-webdataset
    caption_dataset = load_dataset('CaptionEmporium/conceptual-captions-cc12m-llavanext')['train']
    txts = raw_dataset['txt']
    captions = caption_dataset['caption']

    caption_dict = {caption: index for index, caption in enumerate(captions)}

    # caption_llava = caption_dataset['caption_llava']
    # caption_llava_dict = {caption: index for index, caption in enumerate(caption_llava)}
    #
    # caption_llava_short = caption_dataset['caption_llava_short']
    # caption_llava_short_dict = {caption: index for index, caption in enumerate(caption_llava_short)}

    # from tqdm import tqdm
    # for example in tqdm(raw_dataset):
    #     txt = example['txt']
    #     if txt in caption_dict:
    #         caption_index = caption_dict[txt]
    #         caption_llava = caption_dataset[caption_index]['caption_llava']
    #         caption_llava_short = caption_dataset[caption_index]['caption_llava_short']
    #         caption = caption_dataset[caption_index]['caption']

    def process_example(example):
        webp_image = resize_and_convert_to_webp(example['jpg'])
        txt = example['txt']
        # if txt in caption_dict:
        if webp_image is not None:
            return {'webp': webp_image, 'key': example['__key__'], 'txt': txt}
        else:
            return {}

    processed_dataset = raw_dataset.map(
        process_example,
        remove_columns=raw_dataset.column_names,
        num_proc=128,
        batched=False
    )

    processed_dataset.save_to_disk('./cc12m_512')














from datasets import load_from_disk
from huggingface_hub import HfApi
new_dataset = load_from_disk('./cc12m_512')
api = HfApi()
api_token = "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"

api.upload_folder(
    folder_path='./cc12m_512',
    repo_id="nebula/cc12m",
    repo_type="dataset",
    token=api_token
)




import os
import numpy as np
from data import load_dataset,config
dataset = load_dataset('hammh0a/SynthCLIP')


pixparse/cc3m-wds












import os
import numpy as np
from data import load_dataset
os.environ['HF_HOME'] = '/data/jwang/yb/cache'
dataset = load_dataset('poloclub/diffusiondb', '2m_all')




# CaptionEmporium/conceptual-captions-cc12m-llavanext



import torch
from torch.utils.data import Dataset
from data import load_dataset
raw_dataset = load_dataset('justram/yfcc100m_openai_subset')





export HF_HOME=/data/jwang/yb/cache


import os
import numpy as np
from data import load_dataset,config
dataset = load_dataset('drawthingsai/megalith-10m')



drawthingsai/megalith-10m



# JourneyDB/JourneyDB









