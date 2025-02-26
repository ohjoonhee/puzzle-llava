import torch
import json
import torch.utils
import torch.utils.cpp_extension
import tqdm
from PIL import Image
import random
from PIL import ImageDraw, ImageSequence
import sliding_puzzles


if __name__ == "__main__":
    # with open("playground/data/patch_swapped_10k_processed.json") as f:
    #     items = json.load(f)

    # print(items[0])
    img = "assets/extreme-ironing-taxi-2.jpg"
    img = Image.open(img).convert("RGB")
    print(img.size)

    env = sliding_puzzles.make(
        w=3,
        variation="image",
        image_folder="/mnt/ssd/Projects/LLaVA/assets",
        image_pool_size=1,
        render_mode="rgb_array",
        image_size=(642, 350),
        shuffle_mode="serial",
        shuffle_render=True,
    )
    obs, infor = env.reset()
    print(obs.shape)
