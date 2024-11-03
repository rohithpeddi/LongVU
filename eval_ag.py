import os
import numpy as np
import torch
from tqdm import tqdm

from longvu.builder import load_pretrained_model
from longvu.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from longvu.conversation import conv_templates, SeparatorStyle
from longvu.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from decord import cpu, VideoReader

class VideoCaptionGenerator:
    def __init__(self, model_path, model_name):
        # Initialize the model and tokenizer
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, model_name,
        )
        self.model.eval()
        self.device = self.model.device

    def generate_caption(self, video_path, question="Describe this video in detail"):
        # Read the video frames
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        fps = float(vr.get_avg_fps())
        frame_indices = np.array([i for i in range(0, len(vr), round(fps))])
        video_frames = []
        for frame_index in frame_indices:
            img = vr[frame_index].asnumpy()
            video_frames.append(img)
        video_frames = np.stack(video_frames)
        image_sizes = [frame.shape[:2] for frame in video_frames]
        video_processed = process_images(video_frames, self.image_processor, self.model.config)
        video_processed = [item.unsqueeze(0) for item in video_processed]

        # Prepare the prompt
        qs_input = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv = conv_templates["qwen"].copy()
        conv.append_message(conv.roles[0], qs_input)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize the input
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        # Generate the caption
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=video_processed,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.2,
                max_new_tokens=384,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        pred = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return pred

    def process_video_directory(self, video_dir, output_dir, question):
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Get list of video files in the directory
        video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

        # Iterate over each video file
        for video_file in tqdm(video_files):
            video_path = os.path.join(video_dir, video_file)
            print(f"Processing video: {video_file}")
            # Generate caption
            caption = self.generate_caption(video_path, question=question)
            # Write the caption to a separate text file
            base_name = os.path.splitext(video_file)[0]
            output_file = os.path.join(output_dir, f"{base_name}_caption.txt")
            with open(output_file, 'w') as f_out:
                f_out.write(caption)
            print(f"Caption for {video_file} written to {output_file}\n")


def main():
    model_path = "/data/rohith/checkpoints/LongVU_Qwen2_7B"
    model_name = "cambrian_qwen"
    video_dir = "/data/rohith/ag/videos"
    output_dir = "/data/rohith/ag/captions/longvu"
    os.makedirs(output_dir, exist_ok=True)
    question = ("Describe this video in detail particularly focusing on actions "
                "performed by the persons present in the video and their respective interactions with objects.")

    generator = VideoCaptionGenerator(model_path, model_name)
    generator.process_video_directory(video_dir, output_dir, question=question)

if __name__ == "__main__":
    main()