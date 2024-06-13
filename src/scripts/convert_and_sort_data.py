import pandas as pd, os, os.path, ast, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split('/')[:-2])
sys.path.append(parent_dir)
from src.data import loading


def _process_image(src_dir: str, dst_dir: str, line: pd.Series, category_name: str, category_index: int):
    src_name = line['image'] + '.png'
    dst_name = f'{category_index}/{src_name}'
    dcm_name = '/'.join(line['dicom'].split('/')[-2:])

    view = line['view']
    horizontal_flip = line['horizontal_flip']
    best_center = line['best_center'][view][0]

    image = loading.read_image(os.path.join(src_dir, src_name), dtype=None)
    image = loading.flip_and_crop(image, view, horizontal_flip, best_center)
    loading.write_image(os.path.join(dst_dir, dst_name), image)
    return dst_name, dcm_name, category_name


def convert_vindr_mammo_dataset_to_our_storage_format():
    parent_path = '/home/student/sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output'
    image_dir = os.path.join(parent_path, 'cropped_images')
    dict_path = os.path.join(parent_path, 'dictionary.csv')
    top_c = 6

    src = pd.read_csv(dict_path, converters={'best_center': ast.literal_eval, 'finding_categories': ast.literal_eval})
    dst = pd.DataFrame({'png': [], 'dicom': [], 'label': []}, dtype=str)

    # Delete images which don't have exactly one category
    for i in range(len(src)):
        if len(src.loc[i, 'finding_categories']) != 1:
            src.drop(i, inplace=True)
        else:
            src.loc[i, 'finding_categories'] = src.loc[i, 'finding_categories'][0]

    src.reset_index(inplace=True)
    categories = list(src['finding_categories'].value_counts()[:top_c].keys())

    dest_path = '/home/student/gmic/vindrmammo_data'
    for index, category in enumerate(categories):
        os.makedirs(os.path.join(dest_path, str(index)), exist_ok=True)

        for line in src[src['finding_categories'] == category].iloc:
            png_name, dicom_name, label_name = _process_image(image_dir, dest_path, line, category, index)
            dst.loc[len(dst), :] = [png_name, dicom_name, label_name]

    dst.to_csv(os.path.join(dest_path, 'mapping.csv'))


if __name__ == "__main__":
    convert_vindr_mammo_dataset_to_our_storage_format()
