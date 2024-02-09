import numpy as np
import cv2
import os
import random
import torch 
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import argparse

from io import BytesIO
from PIL import Image
import numpy as np
pylab.rcParams['figure.figsize'] = 20, 12
import timeit
from tqdm import tqdm

import clip

def get_video_clips(video_path, video_length):

    # get video basic info
    video_name = os.path.splitext(os.path.basename(input_dir + video_info))[0] 
    video_cap  = cv2.VideoCapture(video_path)
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    
    # output for numpy arr of frames (N, H, W, C)
    frames = []
    num_frames = video_length * fps
    success = True
    count = 0
    while success and count < num_frames:
        success, img = video_cap.read()
        if success:
            frames.append(img)
            count += 1
    # video_frames = np.stack(frames, axis=0) # (N, H, W, C)
    # print(video_frames.shape)
    video_cap.release()
    cv2.destroyAllWindows()
    return frames, frame_width, frame_height, video_name, fps

def get_full_video_clips(video_path):

    # get video basic info
    video_name = os.path.splitext(os.path.basename(input_dir + video_info))[0] 
    video_cap  = cv2.VideoCapture(video_path)
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    frames_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('video_name: {}\tframe_count: {}\tfps: {}'.format(video_name, int(frames_count), fps))
    
    # output for numpy arr of frames (N, H, W, C)
    frames = []
    success = True
    count = 0
    while success and count < frames_count:
        success, img = video_cap.read()
        if success:
            color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(color_coverted)
            frames.append(img)
            count += 1
    # video_frames = np.stack(frames, axis=0) # (N, H, W, C)
    # print(frames_count)
    # print(len(frames))
    video_cap.release()
    cv2.destroyAllWindows()
    # pdb.set_trace()
    return frames, frame_width, frame_height, video_name, fps, frames_count


if __name__ == '__main__':
    input_dir_nlq = '/oscar/data/csun45/czhan164/QAEgo4D/v1/clips/'
    output_dir_nlq = '/oscar/data/csun45/czhan164/QAEgo4D/v1/clips_embedding/'
    
    input_dir = '/gpfs/data/csun45/cfu17/ego4d_fho_data/v1/clips_low_res/'
    output_dir = '/gpfs/data/csun45/cfu17/GLIP_temp/output_CLIP_img_embedding_kmeans_ego4d/'
    target_fps = 1
    target_out_fps = 30
    
    input_dir_50salads = '/gpfs/data/csun45/cfu17/50salads/rgb/'
    output_dir_50salads = '/gpfs/data/csun45/cfu17/GLIP_temp/output_embedding_50salads_v2/'
    output_dir_img_50salads = '/gpfs/data/csun45/cfu17/GLIP_temp/output_CLIP_img_embedding_50salads/'
    
    input_dir_breakfast = '/gpfs/data/csun45/cfu17/GLIP/Breakfast/'
    output_dir_breakfast = '/gpfs/data/csun45/cfu17/GLIP_temp/output_embedding_breakfast/'
    output_dir_img_breakfast = '/gpfs/data/csun45/cfu17/GLIP_temp/output_CLIP_img_embedding_breakfast/'
    
    input_dir_gaze = '/users/cfu17/data/cfu17/EGTEA Gaze+/gaze_videos_cut/'
    output_dir_gaze = '/gpfs/data/csun45/cfu17/GLIP_temp/output_CLIP_img_embedding_gaze/'
    
    input_dir_tvhi = '/gpfs/data/csun45/cfu17/TVHI/tv_human_interactions_videos/'
    output_dir_tvhi = '/gpfs/data/csun45/cfu17/GLIP_temp/output_CLIP_img_embedding_tvhi/'
    output_dir_tvhi2 = '/gpfs/data/csun45/cfu17/GLIP_temp/output_CLIP_img_embedding_tvhi_24fps/'
    
    input_dir_ego4D_v2 = '/gpfs/data/csun45/cfu17/ego4d_fho_data_v2/v2/clips_low_res/'
    output_dir_ego4D_v2 = '/gpfs/data/csun45/cfu17/GLIP_temp/output_embedding_ego4dv2_clip_olcs/'
    
    input_dir_ek55 = '/users/swang299/data/cfu17/GLIP/EK_lowres_cut/ek55/'
    output_dir_ek55 = '/gpfs/data/csun45/cfu17/GLIP_temp/output_CLIP_img_embedding_ek55/'
    
    input_dir = input_dir_ek55
    output_dir = output_dir_ek55
    
    parser = argparse.ArgumentParser(description='A test program.')
    # parser.add_argument("--embed_type",help="Embedding_type for generation", default="obj", type=str)
    parser.add_argument("--start_idx", help="Start idx for generation", default=0, type=int)
    parser.add_argument("--end_idx", help="End idx for generation", default=0, type=int)
    parser.add_argument("--gpu", type=int, default="0")
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu) 
    start_idx = args.start_idx
    end_idx = args.end_idx
    
    # print(clip.available_models())
    # pdb.set_trace()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    listing = os.listdir(input_dir)
    # remove files that are not mp4
    listing = [x for x in listing if x.endswith(".mp4")]
    if end_idx == 0:
        listing = listing[start_idx:]
        print("listing ID start: ", listing[0], len(listing))
        print("listing idx from {} to end".format(start_idx))
    else:
        listing = listing[start_idx:end_idx]
        print("listing ID start: ", listing[0], len(listing))
        print("listing idx from {} to {}".format(start_idx, end_idx))
    num_video = len(listing)
    
    all_video_num = len(listing)

    tic = timeit.time.perf_counter()
    video_count = 0
    inference_count = 1
    with tqdm(total=all_video_num) as pbar:
        for video_info in listing:
            if video_count < num_video:
                video_path = input_dir + video_info
                video_name = os.path.splitext(os.path.basename(input_dir + video_info))[0] 
                out_path = output_dir + video_name + '.pt'

                # loading video clips into frames
                video_frames, frame_width, frame_height, video_name, fps, frames_count= get_full_video_clips(video_path)
                
                if os.path.exists(out_path):
                    # print("{} already exists".format(out_path))
                    video_count += 1
                    pbar.update(1)
                    continue
                
                # video_frames, frame_width, frame_height, video_name, fps = get_video_clips(video_path, 60)
                # frames_count = 1800
                # calculate stride 
                stride = int(int(fps) / target_fps)
                
                # small fix for rounding
                if stride == 0:
                    stride = 1
                # output video 
                out_file_all = torch.zeros((int(frames_count), 1, 768)) # (C,10,256+4+1+1)
                
                # CLIP embedding
                image_input = preprocess(video_frames[0]).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    
                for idx, img in enumerate(video_frames):
                    if idx % stride == 0 and idx != 0:
                        # print(idx)
                        image_input = preprocess(img).unsqueeze(0).to(device)

                        with torch.no_grad():
                            image_features = model.encode_image(image_input)
                        
                        inference_count += 1
                        # print(predictions.get_field("labels"))
                        # print(predictions.get_field("scores"))
                        
                    # edge case when predicted box less than 10
                    out_file_all[idx] = image_features
                    # print(out_file.shape) # [9, 262]
                    # print(out_file_all[idx].shape) # [10, 262]
                    # print(predictions.bbox.shape) # xyxy                        
                # print("frames_count: ", frames_count)
                # print("inference_count: ", inference_count)
                # print("frame_width: ", frame_width)
                # print("frame_height: ", frame_height)
                # print("Video_frames: ", frames_count)
                # print("Length of video: {}. Number of bounding box embeddings: {} Shape of embedding: {}".format(len(out_file), len(out_file[0]), out_file[0][0].shape))
                # print("out tensor shape: ", out_file_all.shape)
                out_file_final = torch.squeeze(out_file_all)
                # print("final out tensor shape: ", out_file_final.shape)
                # pdb.set_trace()
                torch.save(out_file_final, out_path)
                video_count += 1
                pbar.update(1)
    print("Embedding time for {} videos with {} pred_fps: {}".format(num_video, target_fps, timeit.time.perf_counter() - tic))
