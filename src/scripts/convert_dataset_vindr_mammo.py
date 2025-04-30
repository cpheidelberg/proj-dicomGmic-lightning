import pandas as pd
import os
import ast
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = '/'.join(current_dir.split('/')[:-2])
sys.path.append(parent_dir)

from src.data_loading import loading


"""
This script reads data from the VindrMammo dataset and converts it to
the format we use in data_loading.dataset.ClassificationImages.

Input format:
- directory `cropped_images` containing PNG files, unprocessed
- CSV file `dictionary.csv` with information about the images

Output format:
- root directory `vindrmammo_data`
    - subdirectories `0`, `1`, ..., `K-1` for the top K classes (here, K = 6)
        Each subdirectory contains PNGs representing scans belonging to
        the class. The PNGs are preprocessed, prepared to be given to the GMIC.
    - CSV file `mapping.csv` which has one line per image with three columns:
        - png: file path of the PNG image, relative to vindrmammo_data (e.g. "2/1357.png")
        - dicom: file path of the original DICOM file
        - label: string representing the class (e.g. "Suspicious Calcification")
"""


def process_image(src_dir: str, dst_dir: str, line: pd.Series, category_name: str, category_index: int, index: int):
    src_name = line['image'] + '.png'
    dst_name = f'{category_index}/{index}.png'
    dcm_name = '/'.join(line['dicom'].split('/')[-2:])

    image = loading.read_image(os.path.join(src_dir, src_name))
    image = loading.adjust_brightness(image)
    image = loading.flip_image(image, line['view'], line['horizontal_flip'])
    image = loading.crop_image(image, line['view'], line['best_center'][line['view']][0])

    loading.write_image(os.path.join(dst_dir, dst_name), image)
    return [dst_name, dcm_name, category_name]


def main():
    src_dir = '/home/ubuntu/sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/cropped_images'
    dictionary_path = '/home/ubuntu/sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/dictionary.csv'
    top_c = 6

    dctn = pd.read_csv(dictionary_path, converters={'best_center': ast.literal_eval, 'finding_categories': ast.literal_eval})
    mapp = pd.DataFrame({'png': [], 'dicom': [], 'label': []}, dtype=str)

    # Delete images which don't have exactly one category
    for i in range(len(dctn)):
        if len(dctn.loc[i, 'finding_categories']) != 1:
            dctn.drop(i, inplace=True)
        else:
            dctn.loc[i, 'finding_categories'] = dctn.loc[i, 'finding_categories'][0]

    dctn.reset_index(inplace=True)
    categories = list(dctn['finding_categories'].value_counts()[:top_c].keys())

    dst_dir = '/home/ubuntu/gmic/vindrmammo_data'
    counter = 0
    for category_index, category_name in enumerate(categories):
        os.makedirs(os.path.join(dst_dir, str(category_index)), exist_ok=True)

        for i, line in enumerate(dctn[dctn['finding_categories'] == category_name].iloc):
            mapp.loc[len(mapp), :] = process_image(src_dir, dst_dir, line, category_name, category_index, i)

            counter += 1
            print(f'#{counter} - {100*counter//len(dctn)} %')

    mapp.to_csv(os.path.join(dst_dir, 'mapping.csv'))


if __name__ == "__main__":
    main()
