from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import torch
from torchvision.ops import box_convert
import torchvision.transforms as TT
import numpy as np
from transformers import SamModel, SamProcessor
import matplotlib.pyplot as plt
from PIL import Image
import os,jsonlines,io,decord,random
import webdataset as wds
from decord import VideoReader,cpu

def is_relevant_category(word):
    relevant_categories = ['animal', 'vehicle', 'person', 'plant']
    unrelevant_categories = ['body', 'clothing']
    irrelevant_words = {
        "background", "scene", "situation", "architecture", "hand", "hair", "face", "head", "eye", "ear", "nose", "mouth", "skin", "front",
        "shirt", "pants", "dress", "jacket", "coat", "suit", "blouse", "sweater", "jeans", "shorts", "t-shirt", "socks", "skirt", "tie"
    }
    color_words = {"blue", "red", "green", "yellow", "black", "white", "gray", "orange", "purple", "pink", "brown"}
    
    if word.lower() in irrelevant_words or word.lower() in color_words:
        return False

    synsets = wordnet.synsets(word)
    for synset in synsets:
        if any(category in synset.lexname() for category in unrelevant_categories):
            print(f'{word} in {synset.lexname()}')
            return False
        elif any(category in synset.lexname() for category in relevant_categories):
            if synset.lexname() not in {'noun.state', 'noun.cognition', 'noun.time'}:
                print(f'{word}:  {synset.lexname()}')
                return True
    return False

def get_subjects_from_prompt(prompt):
    tokens = word_tokenize(prompt)

    tagged_tokens = pos_tag(tokens)

    relevant_tags = {"NN", "NNS"}
    filtered_tokens = [word for word, tag in tagged_tokens if tag in relevant_tags]

    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in filtered_tokens if word.lower() not in stop_words]

    lemmatizer = WordNetLemmatizer()
    subject_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    relevant_words = [word for word in subject_words if is_relevant_category(word)]

    unique_subject_words = list(dict.fromkeys(relevant_words))

    subject_sentence = " . ".join(unique_subject_words)
    return subject_sentence

class Grounding:
    """
    accept image tensor and than grounding subject by reference words
    """
    def __init__(self, config, model_dir):
        
        self.model = load_model(config, model_dir)
        self.BOX_TRESHOLD = 0.4
        self.TEXT_TRESHOLD = 0.25
        self.transform = TT.Compose([
            TT.ToPILImage(), 
            TT.Resize([800], max_size=1333),
            TT.ToTensor(),
            TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def transform_image(self, frames):
        return self.transform(frames)
    
    def get_boxes(self, image, subject_words):
        
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=subject_words,
            box_threshold=self.BOX_TRESHOLD,
            text_threshold=self.TEXT_TRESHOLD
        )

        return boxes, logits, phrases
    
    def process_boxes(self, image_source, boxes):
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        result = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        result = result.unsqueeze(0).tolist() ## the segment accept this shape

        return result
    
class Segment:
    def __init__(self, model):
        self.model = SamModel.from_pretrained(model).to("cuda")
        self.processor = SamProcessor.from_pretrained(model)

    def get_masks(self, image_RGB, boxes):

        inputs = self.processor(image_RGB, input_boxes=boxes, return_tensors="pt").to("cuda")
        outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        scores = outputs.iou_scores
        print(scores)

        return masks, scores

def drwa_mask_image(image_RGB, masks, phrases):
    for i,mask in enumerate(masks[0]):
        mask_array = mask.numpy().astype(np.uint8)
        mask_array = mask_array * 255

        if mask_array.ndim == 4:
            mask_array = mask_array[0, 0, :, :]
        elif mask_array.ndim == 3:  # (1, H, W)
            mask_array = mask_array[0, :, :]

        result_image = np.zeros_like(image_RGB.numpy())
        result_image[mask_array == 255] = image_RGB[mask_array == 255]
        mask_image = Image.fromarray(result_image)  # 转换为 PIL 图像
        mask_image.save(f"Segment/{phrases[i]}.png")  # 保存文件
        print(f"Mask {i} only image saved as mask_only_{i}.png")



def get_image_from_tar(tar_path):
    sadfa =1

def find_all_tar(tar_path):
    tar_all = []
    for dirpath, _, filenames in os.walk(tar_path):
        for filename in filenames:
            if filename.endswith('.tar'):
                tar_all.append(os.path.join(dirpath,filename))

    return tar_all

def get_prompt(tars):
    prompt = {}
    for tar in tars:
        try:
            meta_name = tar.replace('.tar', '.meta.jsonl')
            meta_data = jsonlines.open(meta_name)

            for contents in meta_data:
                prompt[contents['key']] = contents['long_caption']
        except:
            continue
    return prompt

def GSAM_main(tar_path, skip_frms_num = 3):
    transform = TT.Compose([
        TT.ToPILImage(), 
        TT.Resize([800], max_size=1333),
        TT.ToTensor(),
        TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tars = find_all_tar(tar_path)[:5]
    prompt = get_prompt(tars)

    url = wds.SimpleShardList(tars)
    pipeline = wds.DataPipeline(
        url,
        wds.split_by_node,
        wds.split_by_worker, 
        wds.tarfile_to_samples(),
        wds.decode("pil"),
        wds.to_tuple("__key__", "mp4"),
    )

    Grounding_Task = Grounding("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth")
    Segment_Task = Segment("/workspace/Crilias/zhangzhenxing/ckpt/sam-vit-base")

    for key, mp4 in pipeline:
        try:
            long_caption = prompt[key]
            subject_sentence = get_subjects_from_prompt(long_caption)
        except KeyError:
            print(f"Warning: Missing prompt for key {key}. Skipping this key.")
            continue
        decord.bridge.set_bridge('torch')

        if mp4 is not None:
            decord_vr = VideoReader(io.BytesIO(mp4), ctx=cpu(0))
        else:
            print(f"VideoReader initialization failed for key {key}")
            continue

        total_frames = len(decord_vr)

        start = skip_frms_num
        end = total_frames - skip_frms_num

        select_frame = random.randint(start, end)
        origin_frms = decord_vr.get_batch([select_frame])[0]

        temp_frms = origin_frms.permute(2, 0, 1)

        image_transformed = transform(temp_frms)

        boxes, logits, phrases = Grounding_Task.get_boxes(image_transformed, subject_sentence)
        boxes = Grounding_Task.process_boxes(origin_frms, boxes)

        masks,scores = Segment_Task.get_masks(origin_frms, boxes=boxes)

        drwa_mask_image(origin_frms, masks,phrases)
        break

if __name__ =="__main__":
    GSAM_main('/workspace/Data/video_and_image_datas/processed_videos_wds_datas/hd_movie_video_data')


        




        

    
