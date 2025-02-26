import os
import os.path as osp
import torch
import json
import tqdm
import numpy as np
import argparse

from PIL import Image

import glob


def patch_mix(img, patch_perm):
    img_array = np.array(img)
    # Get the dimensions of the image
    if len(img_array.shape) == 3:
        height, width, _ = img_array.shape
    elif len(img_array.shape) == 2:
        height, width = img_array.shape
    else:
        raise ValueError("Invalid image shape")
    # Calculate the size of each patch
    patch_height = height // 3
    patch_width = width // 3

    # Create a list to store the patches
    patches = []

    # Extract the patches
    for i in range(3):
        for j in range(3):
            patch = img_array[i * patch_height : (i + 1) * patch_height, j * patch_width : (j + 1) * patch_width]
            patches.append(patch)

    # Shuffle the patches
    patches = np.array(patches)[np.array(patch_perm) - 1]

    # Create a new array to store the shuffled image
    shuffled_img_array = np.zeros_like(img_array)

    # Place the shuffled patches into the new array
    for i in range(3):
        for j in range(3):
            shuffled_img_array[i * patch_height : (i + 1) * patch_height, j * patch_width : (j + 1) * patch_width] = patches[i * 3 + j]

    # Convert the shuffled array back to an image
    shuffled_img = Image.fromarray(shuffled_img_array)

    return shuffled_img


def get_patch_permutation(grid_size=(3, 3)):
    # Get the permutation of the patches
    ADJ_DELTA = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def out_of_bound(x, y, grid_size):
        gridh, gridw = grid_size
        return x < 0 or x >= gridw or y < 0 or y >= gridh

    def cvt2d_to_1d(x, y, grid_size):
        return y * grid_size[1] + x + 1

    gridh, gridw = grid_size
    grid = [[i * gridw + j + 1 for j in range(gridw)] for i in range(gridh)]
    pivy, pivx = np.random.randint(gridh), np.random.randint(gridw)
    adj = ADJ_DELTA[np.random.randint(4)]
    if out_of_bound(pivx + adj[0], pivy + adj[1], grid_size):
        adj = (-adj[0], -adj[1])

    oracle_answer = (cvt2d_to_1d(pivx, pivy, grid_size), cvt2d_to_1d(pivx + adj[0], pivy + adj[1], grid_size))
    oracle_answer = sorted(oracle_answer)
    perm = list(range(1, gridh * gridw + 1))
    perm[oracle_answer[0] - 1], perm[oracle_answer[1] - 1] = perm[oracle_answer[1] - 1], perm[oracle_answer[0] - 1]

    return oracle_answer, perm


def seed_everything():
    np.random.seed(0)


def main(args: argparse.Namespace):
    seed_everything()
    OUTPUT_IMAGE_DIR = "playground/data/pswap_images_train_3x3_50k"
    OUTPUT_JSON_PATH = "playground/data/pswap_annot/train_3x3_50k.json"

    # FIXME: Remove hardcoded lines
    # OUTPUT_IMAGE_DIR = args.image_dir
    # OUTPUT_JSON_PATH = args.output_path

    images = glob.glob("playground/data/coco/train2017/*.jpg")

    # keys: id (string), image (string), conversations (list of dict), oracle_answer (list), grid_size (list of int)
    GRID_SIZE = [3, 3]

    annotations = []

    for idx, img_path in enumerate(tqdm.tqdm(images[:50000]), 1):
        img = Image.open(img_path)
        oracle_answer, perm = get_patch_permutation()
        shuffled_img = patch_mix(img, perm)

        img_basename = osp.basename(img_path)
        shuffled_img.save(osp.join(OUTPUT_IMAGE_DIR, img_basename))

        # PROMPT = f'The image is a patch-swapped version of the original image, divided into a {GRID_SIZE[1]}x{GRID_SIZE[0]} grid with {GRID_SIZE[0]*GRID_SIZE[1]} patches numbered from 1 to {GRID_SIZE[0]*GRID_SIZE[1]} (left to right, top to bottom). Two adjacent patches have been swapped. Analyze the image carefully and provide the answer in the format: "The answer is (swapped patch number, swapped patch number).".'
        PROMPT = f'The image is a patch-swapped version of the original image, divided into a {GRID_SIZE[1]}x{GRID_SIZE[0]} grid with {GRID_SIZE[0]*GRID_SIZE[1]} patches numbered from 1 to {GRID_SIZE[0]*GRID_SIZE[1]} (left to right, top to bottom). Two adjacent patches have been swapped. Analyze the image carefully and provide the answer in the following concise format: "(patch number, patch number).". the correct output would be "(1, 2)". Ensure accuracy and consistency in formatting the output.'

        annotations.append(
            {
                "id": f"{idx:012d}",
                "image": img_basename,
                "conversations": [{"from": "human", "value": f"<image>\n{PROMPT}"}, {"from": "gpt", "value": f"The answer is ({oracle_answer[0]}, {oracle_answer[1]})."}],
                "oracle_answer": oracle_answer,
                "grid_size": GRID_SIZE,
            }
        )

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(annotations, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch Swap Data Constructor")
    parser.add_argument("--image-dir", type=str, help="Directory containing input images")
    parser.add_argument("--output-path", type=str, help="Path to save the output JSON file")
    args = parser.parse_args()

    main(args)
