import torch
import os
import pandas as pd
from torchvision import transforms
import shutil
import random

IMAGE_SIZE = 224


def get_transforms(dataset: str = 'arctic'):
    if dataset == 'arctic':
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop((800, 1000)),  # remove bottom digits in some pictures
                transforms.RandomResizedCrop(IMAGE_SIZE),
                transforms.ColorJitter(brightness=.2, contrast=.2, hue=0),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation((0, 20)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(IMAGE_SIZE + 32),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])}
    elif dataset in ['iNaturalist', 'clip']:
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.1, 1.0)),
                transforms.ColorJitter(brightness=.2, contrast=.2, hue=.05),
                transforms.RandomRotation(30),  # TODO: maybe different angle (30), cutout! (albumentation), cutmix
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),

            'test': transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])}
    else:
        print(f'Stage {dataset} is not defined')
        data_transforms = None
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    return data_transforms


def get_data(root: str, img_dir: str, img_format: str = '.jpeg'):
    df = pd.DataFrame()
    for label in os.listdir(os.path.join(root, img_dir)):
        if os.path.isdir(os.path.join(root, img_dir, label)):
            df_i = pd.DataFrame(
                {'img_path': [x for x in os.listdir(os.path.join(root, img_dir, label)) if x.endswith(img_format)],
                 'label': int(label)})
            df_i['img_path'] = df_i['img_path'].apply(lambda x: os.path.join(label, x))
            # df_i['img_path'] = df_i['img_path'].apply(lambda x: os.path.join(img_dir, label, x))
            df = pd.concat([df, df_i])
    # num_classes = df['label'].nunique()
    num_classes = len(os.listdir(os.path.join(root, img_dir)))
    df.reset_index(drop=True, inplace=True)
    return df, num_classes


def train_test_split_k_shot(df: pd.DataFrame, k: int, num_of_exp: int, test_imgs_per_class_cpu: int = 4):
    if k:
        train = stratified_sample_df(df, 'label', k, num_of_exp)
        # indices = list(map(int, train.index))
        test = df.iloc[[x for x in df.index if x not in train.index]]
        if not torch.cuda.is_available():
            test = stratified_sample_df(test, 'label', test_imgs_per_class_cpu)  # for cpu to train faster
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        return train, test
    else:
        return df, df


def stratified_sample_df(df: pd.DataFrame, col: str, n_samples, random_state: int = 42):
    n = min(n_samples, df[col].value_counts().min())
    if n < n_samples:
        print(f'Too big k. Used {n} samples instead {n_samples} for sampling test.')
    df_ = df.groupby(col).apply(lambda x: x.sample(n, random_state=random_state))
    df_.index = df_.index.droplevel(0)
    return df_


def create_sim2arctic_from_inaturalist(root: str = '.',
                                   old_data_dir: str = 'data/train_mini',
                                   new_data_dir: str = 'data/sim2arctic',
                                   class_size: int = 50):
    # family_genus_species
    botanic_names = {'empty_slot': '__No_plant__',
                     'pepper': 'Solanaceae_Capsicum_',
                     'tomato': 'Solanaceae_Solanum_lycopersicum',
                     'kohlrabi': 'Brassicaceae_Brassica_',  # not really similar to arctic one
                     'frisee': 'Asteraceae_Lactuca_',  # wild, need Lactuca_sativa, not great in clip
                     'lactúca': '__Repeats__', # 'Asteraceae_Lactuca_'
                     'lettuce_oakleaf': '__Repeats__', # 'Asteraceae_Lactuca_'
                     'radish': 'Brassicaceae_Raphanus_raphanistrum',
                     'basil': 'Lamiaceae_Perilla',  # family, no genus
                     'cilantro': 'Apiaceae_Daucus',  # family, no genus
                     'cress': 'Brassicaceae_Lepidium_',
                     'mint': 'Lamiaceae_Lamium_',
                     'chard': 'Amaranthaceae_Beta_vulgaris',  # like beetroot
                     'brassica': '__Repeats__',  # Brassicaceae_Brassica_
                     'lettuce_endivia': '__Repeats__', # 'Asteraceae_Lactuca_'
                     'chives': 'Amaryllidaceae_Allium_schoenoprasum',  # garlic, with flowers often
                     'parsley': 'Apiaceae_Osmorhiza'  # family, no genus
                     }

    arctic_names2labels = {'empty_slot': 0,  # steel table
                           'pepper': 1,
                           'tomato': 2,
                           'kohlrabi': 3,
                           'frisee': 4,  # curly lettuce plants greenhouse, bad in clip
                           'lactúca': 5,
                           'mint': 6,  # mint plants greenhouse
                           'lettuce_oakleaf': 7,  # red oak leaves lettuce plants greenhouse
                           'radish': 8,  # not very good in clip
                           'basil': 9,
                           'cilantro': 10,  # curly parsley
                           'cress': 11,
                           'chard': 12,
                           'brassica': 13,  # not very good in clip
                           'lettuce_endivia': 14,  # not very good in clip
                           'parsley': 15,
                           'chives': 16,
                           # 'basil_2': 17,
                           # 'cilantro': 18,
                           # 'lactúca_2': 19
                           }
    # choose folders with plants
    old_root = os.path.join(root, old_data_dir)
    new_root = os.path.join(root, 'data/plants')
    for folder in os.listdir(old_root):
        if '_Plantae_' in folder:
            shutil.move(os.path.join(old_root, folder), os.path.join(new_root, folder))

    # create folder with desired labels ...
    random.seed(0)
    for cat in botanic_names.keys():
        dest_fpath = os.path.join(root, new_data_dir, str(arctic_names2labels[cat]))
        os.makedirs(dest_fpath, exist_ok=True)
        for folder in os.listdir(new_root):
            if botanic_names[cat] in folder:
                for filename in os.listdir(os.path.join(new_root, folder)):
                    src_fpath = os.path.join(new_root, folder, filename)
                    shutil.copy(src_fpath, dest_fpath)
        # ... and balanced sizes of classes
        files = os.listdir(dest_fpath)
        files_rm = random.sample(files, max(0, len(files) - class_size))
        _ = [os.remove(os.path.join(dest_fpath, file)) for file in files_rm]


def urls_from_clip(root='.', data_dir='data/clip'):
    os.makedirs(f'{root}/{data_dir}/txt', exist_ok=True)
    for file in os.listdir(f'{root}/{data_dir}/json'):
        if file.endswith('.json'):
            name = file.split('.json')[0]
            df = pd.read_json(f'{root}/{data_dir}/json/{file}')
            urls = df['url']
            urls.to_csv(f'{root}/{data_dir}/txt/{name}.txt', index=False, header=False)


def imgs_from_url(output_folder="data/clip/sim2arctic_clip", thread_count=64, number_sample_per_shard=200,
                  oom_shard_count=2, url_folder="data/clip/txt/"):
    cmd = f'img2dataset  --output_folder={output_folder} --thread_count={thread_count} ' \
          f'--number_sample_per_shard={number_sample_per_shard} --oom_shard_count={oom_shard_count} ' \
          f'--url_list={url_folder}'
    for name in range(17):
        file = f'{name}.txt'
        os.system(cmd+file)
