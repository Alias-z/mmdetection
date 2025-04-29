import os
import json
import argparse
import random
from datetime import datetime


def combine_coco_jsons(root_dir, sample_ratio=1.0, random_seed=123):
    # Set random seed for reproducibility
    random.seed(random_seed)

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

    dataset_stats = {}

    for dataset_folder in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_folder)
        if not os.path.isdir(dataset_path):
            continue

        dataset_stats[dataset_folder] = {'train': 0, 'val': 0, 'orig_train': 0, 'orig_val': 0}

        for split, folder in [('train', 'train_sahi'), ('val', 'val_sahi')]:
            json_path = os.path.join(dataset_path, folder, 'sahi_coco.json')
            if not os.path.exists(json_path):
                continue

            with open(json_path, 'r') as file:
                data = json.load(file)

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

            # Randomly sample images if sample_ratio < 1.0
            images = data.get('images', [])
            dataset_stats[dataset_folder][f'orig_{split}'] = len(images)

            if sample_ratio < 1.0:
                num_to_sample = max(1, int(len(images) * sample_ratio))
                sampled_images = random.sample(images, num_to_sample)
                # Get sampled image IDs for filtering annotations later
                sampled_image_ids = {img['id'] for img in sampled_images}
            else:
                sampled_images = images
                sampled_image_ids = {img['id'] for img in images}

            # Record stats
            dataset_stats[dataset_folder][split] = len(sampled_images)

            # Handle images
            for image in sampled_images:
                old_image_id = image['id']
                new_image_id = len(combined_data[split]['images']) + 1
                image_id_mapping[split][old_image_id] = new_image_id
                image['id'] = new_image_id
                # Update file_name to include the correct path with the 'train_sahi' or 'val_sahi' folder
                image['file_name'] = os.path.join(dataset_folder, folder, image['file_name'])
                if 'license' in image:
                    image['license'] = license_id_mapping.get(image['license'], image['license'])
                combined_data[split]['images'].append(image)

            # Handle annotations - only include annotations for sampled images
            for annotation in data.get('annotations', []):
                if annotation['image_id'] in sampled_image_ids:
                    annotation['id'] = annotation_id[split]
                    annotation['image_id'] = image_id_mapping[split][annotation['image_id']]
                    annotation['category_id'] = category_id_mapping[annotation['category_id']]
                    combined_data[split]['annotations'].append(annotation)
                    annotation_id[split] += 1

    # Update info
    for split in ['train', 'val']:
        combined_data[split]['info'].update({
            'description': f'Combined {split} dataset (sample ratio: {sample_ratio}, seed: {random_seed})',
            'date_created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'version': '1.0',
            'contributor': 'COCO JSON Combiner Script',
            'sampling_ratio': sample_ratio,
            'random_seed': random_seed
        })

    # Ensure categories are the same for both train and val
    all_categories = combined_data['train']['categories'] + [cat for cat in combined_data['val']['categories'] if cat not in combined_data['train']['categories']]
    combined_data['train']['categories'] = all_categories
    combined_data['val']['categories'] = all_categories

    # Print dataset statistics
    print("\nDataset sampling statistics:")
    for dataset_name, counts in dataset_stats.items():
        print(f"  {dataset_name}: Train={counts['train']}/{counts['orig_train']} images ({counts['train']/max(1, counts['orig_train'])*100:.1f}%), "
              f"Val={counts['val']}/{counts['orig_val']} images ({counts['val']/max(1, counts['orig_val'])*100:.1f}%)")

    for split in ['train', 'val']:
        output_path = os.path.join(root_dir, f'sahi_coco_{split}.json')
        with open(output_path, 'w') as f:
            json.dump(combined_data[split], f, indent=2)
        print(f"\nCombined COCO JSON for {split} saved to: {output_path}")
        print(f"Number of categories in {split}: {len(combined_data[split]['categories'])}")
        print(f"Sampling ratio: {sample_ratio*100:.1f}% (seed: {random_seed})")
        print(f"Total number of images in {split}: {len(combined_data[split]['images'])}")
        print(f"Total number of annotations in {split}: {len(combined_data[split]['annotations'])}")


def main():
    parser = argparse.ArgumentParser(description="Combine COCO JSON files from multiple datasets.")
    parser.add_argument("--root_dir", type=str, default=os.getcwd(),
                        help="Root directory containing the dataset folders. Default is current working directory.")
    parser.add_argument("--sample_ratio", type=float, default=1.0,
                        help="Ratio of images to sample from each dataset (0.0-1.0). Default is 1.0 (use all).")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducible sampling. Default is 42.")
    args = parser.parse_args()

    root_dir = args.root_dir
    sample_ratio = max(0.0, min(1.0, args.sample_ratio))  # Ensure ratio is between 0 and 1
    random_seed = args.random_seed
    combine_coco_jsons(root_dir, sample_ratio, random_seed)


if __name__ == "__main__":
    main()
