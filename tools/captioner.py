from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import io
import base64
import torch
from datasets import load_dataset
import random
import sqlite3
from tqdm import tqdm

# Configuration
MODEL_NAME = "Qwen/Qwen2-VL-72B-Instruct"
DATASET_PATH = './yfcc15m'
DB_PATH = 'example.db'
BATCH_SIZE = 32  # Adjust based on available memory

# Initialize model and processor
device = torch.device("cuda")
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.bfloat16,
# ).to(device)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Load and prepare dataset
dataset = load_dataset(DATASET_PATH)['train']
sample_ids = list(range(len(dataset)))
random.shuffle(sample_ids)

def initialize_database(conn):
    """Create the kv_store table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS kv_store (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    conn.commit()

def photoid_exists(cursor, photoid):
    """Check if a photoid already exists in the database."""
    cursor.execute('SELECT 1 FROM kv_store WHERE key = ?', (photoid,))
    return cursor.fetchone() is not None

def get_base64_image(image_object):
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image_object.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def generate_description(processor, model, messages, device):
    """Generate image description using the model."""
    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.get('input_ids', []), generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
    except Exception as e:
        print(f"Error during generation: {e}")
        return None

def main():
    # Connect to the database
    with sqlite3.connect(DB_PATH) as conn:
        initialize_database(conn)
        cursor = conn.cursor()
        
        # Iterate over the dataset with a progress bar
        for ids in tqdm(sample_ids, desc="Processing samples"):
            try:
                data = dataset[ids]
                photoid = data.get('photoid')
                title = data.get('title', '')
                description = data.get('description', '')
                image_object = data.get('image')
                
                if not photoid or not image_object:
                    continue  # Skip if essential data is missing
                
                if photoid_exists(cursor, photoid):
                    continue  # Skip already processed entries
                
                base64_image = get_base64_image(image_object)
                
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an AI image caption/description expert, please provide precise description for input images to enhance people's understanding of the content. \
                            Employ succinct keywords or phrases, steering clear of elaborate sentences, adjective, and extraneous conjunctions. \
                            Your description should capture key objects and concise and clear.\
                            When tagging photos of people, include specific details like gender, nationality, age, etc. \
                            Recognize any tag any celebrities, well-known landmark or IPs if clearly featured in the image. \
                            Your words should be accurate, non-duplicative, and have a 250 word count range. Direct give me the content, don't start with 'This image' or 'The image'."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": f"data:image;base64,{base64_image}",
                            },
                            {
                                "type": "text",
                                "text": (
                                    f"I will provide the post titles and descriptions for this image, but these are just user inputs and do not mean they are absolutely reliable. "
                                    f"For reference only.\nTitle: {title} \nDescription: {description} \nDescribe this image below"
                                ),
                            }
                        ],
                    }
                ]
                
                output_text = generate_description(processor, model, messages, device)
                if output_text:
                    cursor.execute(
                        '''INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)''',
                        (photoid, output_text[0])
                    )
                    print(output_text)
                    conn.commit()
            except Exception as e:
                print(f"Error processing ID {ids}: {e}")
                continue  # Continue with the next sample even if there's an error

if __name__ == "__main__":
    main()

# import sqlite3

# # Path to your SQLite database
# DB_PATH = 'example.db'

# def print_all_data(db_path):
#     """Print all data from the kv_store table in the SQLite database."""
#     try:
#         # Connect to the database
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()
#         cursor.execute('SELECT * FROM kv_store')
#         rows = cursor.fetchall()
#         # Print the fetched records
#         if rows:
#             print("Data in kv_store table:")
#             for row in rows:
#                 print(f"Key: {row[0]}, Value: {row[1]}")
#         else:
#             print("No data found in kv_store table.")
#     except sqlite3.Error as e:
#         print(f"Error accessing the database: {e}")
#     finally:
#         # Ensure the connection is closed
#         if conn:
#             conn.close()

# print_all_data(DB_PATH)