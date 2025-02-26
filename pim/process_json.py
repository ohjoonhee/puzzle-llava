import json
import copy


def read_json(json_path):
    with open(json_path, "r") as f:
        d = json.load(f)

    for e in d:
        if "000000000532.jpg" in e["image"]:
            print(e)
            break


def write_json(json_path, data):
    OUTPUT_JSON_PATH = "playground/data/pswap_annotations_val_10k_processed.json"
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    processed_data = copy.deepcopy(data)

    for d in processed_data:
        GRID_SIZE = d["grid_size"]
        oracle_answer = d["oracle_answer"]
        PROMPT = f'The image is a patch-swapped version of the original image, divided into a {GRID_SIZE[1]}x{GRID_SIZE[0]} grid with {GRID_SIZE[0]*GRID_SIZE[1]} patches numbered from 1 to {GRID_SIZE[0]*GRID_SIZE[1]} (left to right, top to bottom). Two adjacent patches have been swapped. Analyze the image carefully and provide the answer in the following concise format: "(patch number, patch number).". the correct output would be "(1, 2)". Ensure accuracy and consistency in formatting the output.'
        d["conversations"][0]["value"] = PROMPT
        d["conversations"][1]["value"] = str(tuple(oracle_answer)) + "."
        print(d)
        break

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(processed_data, f)


if __name__ == "__main__":
    JSON_PATH = "playground/data/pperm_annot/train_2x2_10k.json"

    read_json(JSON_PATH)
