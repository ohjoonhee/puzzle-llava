import os
import os.path as osp
import torch
import json
import tqdm
import numpy as np

from PIL import Image

import glob


def patch_perm(img, patch_perm, grid_size=(3, 3)):
    gridh, gridw = grid_size
    img_array = np.array(img)
    # Get the dimensions of the image
    if len(img_array.shape) == 3:
        height, width, _ = img_array.shape
    elif len(img_array.shape) == 2:
        height, width = img_array.shape
    else:
        raise ValueError("Invalid image shape")
    # Calculate the size of each patch
    patch_height = height // gridh
    patch_width = width // gridw

    # Create a list to store the patches
    patches = []

    # Extract the patches
    for i in range(gridh):
        for j in range(gridw):
            patch = img_array[i * patch_height : (i + 1) * patch_height, j * patch_width : (j + 1) * patch_width]
            patches.append(patch)

    # Shuffle the patches
    patches = np.array(patches)[np.array(patch_perm) - 1]

    # Create a new array to store the shuffled image
    shuffled_img_array = np.zeros_like(img_array)

    # Place the shuffled patches into the new array
    for i in range(gridh):
        for j in range(gridw):
            shuffled_img_array[i * patch_height : (i + 1) * patch_height, j * patch_width : (j + 1) * patch_width] = patches[i * gridw + j]

    # Convert the shuffled array back to an image
    shuffled_img = Image.fromarray(shuffled_img_array)

    return shuffled_img


def get_patch_permutation(grid_size=(3, 3)):
    # Get the permutation of the patches
    perm = np.arange(1, grid_size[0] * grid_size[1] + 1)
    perm = np.random.permutation(perm)
    oracle_answer = np.argsort(perm) + 1
    oracle_answer = [int(x) for x in oracle_answer]
    return oracle_answer, perm


def seed_everything():
    np.random.seed(0)


def main():
    seed_everything()
    OUTPUT_IMAGE_DIR = "playground/data/pperm_images/train_3x3_10k"
    OUTPUT_JSON_PATH = "playground/data/pperm_annot/train_3x3_10k.json"

    images = glob.glob("playground/data/coco/train2017/*.jpg")
    print(len(images))

    # keys: id (string), image (string), conversations (list of dict), oracle_answer (list), grid_size (list of int)
    GRID_SIZE = [3, 3]

    annotations = []

    for idx, img_path in enumerate(tqdm.tqdm(images[:10000]), 1):
        img = Image.open(img_path)
        oracle_answer, perm = get_patch_permutation(grid_size=GRID_SIZE)
        shuffled_img = patch_perm(img, perm, grid_size=GRID_SIZE)

        img_basename = osp.basename(img_path)
        shuffled_img.save(osp.join(OUTPUT_IMAGE_DIR, img_basename))

        # PROMPT = f"The image is a patch-swapped version of the original image, divided into a {GRID_SIZE[1]}x{GRID_SIZE[0]} grid with {GRID_SIZE[0]*GRID_SIZE[1]} patches numbered from 1 to {GRID_SIZE[0]*GRID_SIZE[1]} (left to right, top to bottom). Two adjacent patches have been swapped. Analyze the image step by step, explain your reasoning in detail, and finally provide the answer in the format: “The answer is (swapped patch number, swapped patch number)”."
        PROMPT = f'The image is a patch-shuffled version of the original image, divided into a grid of size {GRID_SIZE[1]}x{GRID_SIZE[0]}, resulting in {GRID_SIZE[0]*GRID_SIZE[1]} patches numbered sequentially from 1 to {GRID_SIZE[0]*GRID_SIZE[1]} (from left to right and top to bottom). These patches have been shuffled from their original order and rearranged to form the given image.\n\nYour task is to carefully analyze the image and determine the correct sequential ordering of the patches as they appeared in the original image. Provide your answer in the following format: "(p1, p2, p3, ..., pn)", where each \'pi\' represents the original patch number at that position.\n\nFor example, if the correct order of patches is 3, 1, 2, and 4, the output should be: "(3, 1, 2, 4)". Ensure the output strictly follows this format and reflects the accurate order of patches.'

        annotations.append(
            {
                "id": f"{idx:012d}",
                "image": img_basename,
                "conversations": [{"from": "human", "value": f"<image>\n{PROMPT}"}, {"from": "gpt", "value": str(tuple(oracle_answer)) + "."}],
                "oracle_answer": oracle_answer,
                "grid_size": GRID_SIZE,
            }
        )

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(annotations, f)


if __name__ == "__main__":
    main()
