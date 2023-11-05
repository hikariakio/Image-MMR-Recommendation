import json
import sys

def extract_names_from_json(file_path):
    try:
        # Read the JSON data from the file
        with open(file_path, 'r') as file:
            data_dict = json.load(file)

        # # Extract the list of names from the 'data' attribute
        # captions_with_image_ids_list = [
        #     [item['caption'], item['image_id']]
        #     for item in data_dict['annotations']
        # ]


        #  THIS IS TO REMOVFE DUPLICATES
        seen_ids = set()
        captions_with_image_ids_list = []
        for item in data_dict['annotations']:
            if item['image_id'] not in seen_ids:
                seen_ids.add(item['image_id'])
                captions_with_image_ids_list.append([item['caption'], item['image_id']])

        return captions_with_image_ids_list

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []




# names = extract_names_from_json("val2017/jsonAnnotation.json")
# print("Names from JSON file:", len(names))