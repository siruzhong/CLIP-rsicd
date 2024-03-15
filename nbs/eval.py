import json
import numpy as np
import os
from tqdm import tqdm

from transformers import CLIPProcessor, FlaxCLIPModel
from PIL import Image

# Load RSICD dataset
DATA_DIR = "/hpc2hdd/home/szhong691/zsr/projects/dataset/RSICD"
IMAGES_DIR = os.path.join(DATA_DIR, "RSICD_images")
CAPTIONS_FILE = os.path.join(DATA_DIR, "annotations_rsicd/dataset_rsicd.json")
FEATURES_DIR = os.path.join(DATA_DIR, "features") # Directory to save individual features

# Load model and processor
model_path = "/hpc2hdd/home/szhong691/zsr/projects/UrbanCross_Baselines/CLIP-rsicd/clip-vit-base-patch32"
model = FlaxCLIPModel.from_pretrained(model_path)
processor = CLIPProcessor.from_pretrained(model_path)

# Load RSICD annotations and preprocess
with open(CAPTIONS_FILE, 'r') as f:
    rsicd_data = json.load(f)

# Preprocessing to organize data correctly
image_captions_test = {}
for img in rsicd_data['images']:
    if img['split'] == 'test':  # Check if the split is 'test'
        image_captions_test[img['filename']] = [sentence['raw'] for sentence in img['sentences']]

# Ensure the features directory exists
if not os.path.exists(FEATURES_DIR):
    os.makedirs(FEATURES_DIR)

def get_features_filename(base_dir, identifier, feature_type="image"):
    return os.path.join(base_dir, f"{identifier}_{feature_type}.npz")

def save_feature(feature, filename):
    np.savez_compressed(filename, feature=feature)

def load_feature(filename):
    return np.load(filename)["feature"]

def compute_image_features(image_path, model, processor):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        image_inputs = processor(images=img, return_tensors="jax", padding=True)
        outputs = model.get_image_features(**image_inputs)
    return outputs

def compute_text_features(text, model, processor):
    inputs = processor(text=[text], return_tensors="jax", padding=True)
    outputs = model.get_text_features(**inputs)
    return outputs

def compute_or_load_features(identifier, data_path, model, processor, base_dir, feature_type="image"):
    filename = get_features_filename(base_dir, identifier, feature_type)
    if os.path.exists(filename):
        return load_feature(filename)
    else:
        if feature_type == "image":
            feature = compute_image_features(data_path, model, processor)
        else:  # text
            feature = compute_text_features(data_path, model, processor)
        save_feature(feature, filename)
        return feature

def compute_similarity_scores(image_features, text_features):
    # Normalize features to unit length
    image_features_norm = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
    text_features_norm = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
    
    # Compute dot product (cosine similarity)
    similarity_scores = np.dot(image_features_norm, text_features_norm.T)
    
    return similarity_scores


def compute_retrieval_metrics(similarity_scores, rank_position=0):
    num_queries = len(similarity_scores)
    recalls = {"R@1": 0, "R@5": 0, "R@10": 0}
    med_r_sum = 0
    
    for scores in similarity_scores:
        ranks = np.argsort(-scores)
        med_r_sum += np.where(ranks == rank_position)[0][0] + 1
        
        for k in recalls.keys():
            threshold = int(k.split("@")[1])
            recalls[k] += int(ranks[0] < threshold)
    
    recalls = {k: (v / num_queries) * 100 for k, v in recalls.items()}
    med_r = med_r_sum / num_queries
    
    return recalls, med_r



def main_evaluation(image_captions_test, IMAGES_DIR, model, processor, FEATURES_DIR):
    text_to_image_results = {"R@1": 0, "R@5": 0, "R@10": 0, "MedR": 0}
    image_to_text_results = {"R@1": 0, "R@5": 0, "R@10": 0, "MedR": 0}
    num_images = len(image_captions_test)

    for image_file, captions in tqdm(image_captions_test.items(), desc="Evaluating image and text retrieval (Test Split)"):
        image_path = os.path.join(IMAGES_DIR, image_file)
        image_features = compute_or_load_features(image_file, image_path, model, processor, FEATURES_DIR, "image")

        # Text-to-Image Retrieval
        text_to_image_scores = []
        for caption in captions:
            text_features = compute_or_load_features(caption, caption, model, processor, FEATURES_DIR, "text")
            score = compute_similarity_scores(image_features, text_features)
            text_to_image_scores.append(score)
        
        recalls, med_r = compute_retrieval_metrics(np.array(text_to_image_scores))
        for k in recalls:
            text_to_image_results[k] += recalls[k]
        text_to_image_results["MedR"] += med_r

        # Image-to-Text Retrieval
        image_to_text_scores = []
        for caption in captions:
            text_features = compute_or_load_features(caption, caption, model, processor, FEATURES_DIR, "text")
            score = compute_similarity_scores(image_features, text_features)
            image_to_text_scores.append(score)

        recalls, med_r = compute_retrieval_metrics(np.array(image_to_text_scores), rank_position=0)
        for k in recalls:
            image_to_text_results[k] += recalls[k]
        image_to_text_results["MedR"] += med_r

    # Averaging metrics over all queries
    for k in text_to_image_results.keys():
        text_to_image_results[k] = text_to_image_results[k] / float(num_images)  # Force floating-point division
        image_to_text_results[k] = image_to_text_results[k] / float(num_images)  # Force floating-point division


    # Displaying the results
    print("Text-to-Image Retrieval Metrics (Test Split):")
    for k, v in text_to_image_results.items():
        print(f"{k}: {v:.2f}")

    print("\nImage-to-Text Retrieval Metrics (Test Split):")
    for k, v in image_to_text_results.items():
        print(f"{k}: {v:.2f}")

# Run main evaluation
main_evaluation(image_captions_test, IMAGES_DIR, model, processor, FEATURES_DIR)
