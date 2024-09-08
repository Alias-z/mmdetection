import os
import json
import argparse
from collections import defaultdict
from datetime import datetime

def combine_coco_jsons(root_dir):
    combined_data = {
        'train': {
            'info': {},
            'licenses': [],
            'images': [],
            'annotations': [],
            'categories': []
        },
        'val': {
            'info': {},
            'licenses': [],
            'images': [],
            'annotations': [],
            'categories': []
        }
    }
    image_id_mapping = {'train': {}, 'val': {}}
    annotation_id = {'train': 1, 'val': 1}
    license_id_mapping = {}
    category_id_mapping = {}

    for dataset_folder in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_folder)
        if not os.path.isdir(dataset_path):
            continue

        for split, folder in [('train', 'train_sahi'), ('val', 'val_sahi')]:
            json_path = os.path.join(dataset_path, folder, 'sahi_coco.json')
            if not os.path.exists(json_path):
                continue

            with open(json_path, 'r') as f:
                data = json.load(f)

            # Handle info
            if not combined_data[split]['info']:
                combined_data[split]['info'] = data.get('info', {})
            
            # Handle licenses
            for license in data.get('licenses', []):
                if license['id'] not in license_id_mapping:
                    new_license_id = len(combined_data[split]['licenses']) + 1
                    license_id_mapping[license['id']] = new_license_id
                    license['id'] = new_license_id
                    combined_data[split]['licenses'].append(license)

            # Handle categories
            for category in data.get('categories', []):
                if category['id'] not in category_id_mapping:
                    new_category_id = len(category_id_mapping) + 1
                    category_id_mapping[category['id']] = new_category_id
                    category['id'] = new_category_id
                    combined_data[split]['categories'].append(category)

            # Handle images
            for image in data.get('images', []):
                old_image_id = image['id']
                new_image_id = len(combined_data[split]['images']) + 1
                image_id_mapping[split][old_image_id] = new_image_id
                
                image['id'] = new_image_id
                # Update file_name to include the correct path with the 'train_sahi' or 'val_sahi' folder
                image['file_name'] = os.path.join(dataset_folder, folder, image['file_name'])
                if 'license' in image:
                    image['license'] = license_id_mapping.get(image['license'], image['license'])
                combined_data[split]['images'].append(image)

            # Handle annotations
            for annotation in data.get('annotations', []):
                annotation['id'] = annotation_id[split]
                annotation['image_id'] = image_id_mapping[split][annotation['image_id']]
                annotation['category_id'] = category_id_mapping[annotation['category_id']]
                combined_data[split]['annotations'].append(annotation)
                annotation_id[split] += 1

    # Update info
    for split in ['train', 'val']:
        combined_data[split]['info'].update({
            'description': f'Combined {split} dataset',
            'date_created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'version': '1.0',
            'contributor': 'COCO JSON Combiner Script'
        })

    # Ensure categories are the same for both train and val
    all_categories = combined_data['train']['categories'] + [cat for cat in combined_data['val']['categories'] if cat not in combined_data['train']['categories']]
    combined_data['train']['categories'] = all_categories
    combined_data['val']['categories'] = all_categories

    for split in ['train', 'val']:
        output_path = os.path.join(root_dir, f'sahi_coco_{split}.json')
        with open(output_path, 'w') as f:
            json.dump(combined_data[split], f, indent=2)
        print(f"Combined COCO JSON for {split} saved to: {output_path}")
        print(f"Number of categories in {split}: {len(combined_data[split]['categories'])}")

def main():
    parser = argparse.ArgumentParser(description="Combine COCO JSON files from multiple datasets.")
    parser.add_argument("--root_dir", type=str, default=os.getcwd(),
                        help="Root directory containing the dataset folders. Default is current working directory.")
    args = parser.parse_args()

    root_dir = args.root_dir
    print(f"Processing datasets in: {root_dir}")
    combine_coco_jsons(root_dir)

if __name__ == "__main__":
    main()








