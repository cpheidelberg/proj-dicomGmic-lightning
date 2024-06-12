import pandas as pd, os, ast, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split('/')[:-2])
sys.path.append(parent_dir)
from src.data import loading


def convert_vindr_mammo_dataset_to_our_storage_format():
    parent_path = '/home/student/sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output'
    image_dir = os.path.join(parent_path, 'cropped_images')
    dict_path = os.path.join(parent_path, 'dictionary.csv')
    top_c = 6

    tab = pd.read_csv(dict_path, converters={'best_center': ast.literal_eval, 'finding_categories': ast.literal_eval})
    
    # Delete images which don't have exactly one category
    for i in range(len(tab)):
        if len(tab.loc[i, 'finding_categories']) != 1:
            tab.drop(i, inplace=True)
        else:
            tab.loc[i, 'finding_categories'] = tab.loc[i, 'finding_categories'][0]

    tab.reset_index(inplace=True)
    categories = list(tab['finding_categories'].value_counts()[:top_c].keys())

    dest_path = '/home/student/gmic/vindrmammo_data'
    for index, category in enumerate(categories):
        os.makedirs(os.path.join(dest_path, str(index)), exist_ok=True)

        for line in tab[tab['finding_categories'] == category].iloc:
            image = loading.read_image(f"{os.path.join(image_dir, line['image'])}.png", dtype=None)
            image = loading.flip_and_crop(image, line['view'], line['horizontal_flip'], line['best_center'][line['view']][0])
            loading.write_image(f"{os.path.join(dest_path, str(index), line['image'])}.png", image)

    with open(os.path.join(dest_path, 'labels.txt'), 'w') as labels_file:
        labels_file.write('\n'.join(categories))


if __name__ == "__main__":
    convert_vindr_mammo_dataset_to_our_storage_format()
