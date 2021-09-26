import random
import os
import torch
import numpy as np
from cv2 import imread, resize
from tqdm import tqdm
from collections import Counter
import h5py
import json


def fix_all_seed(seed):
    print('seed-----------all device', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    为训练、验证和测试数据创建输入文件。
    dataset: 数据集名称，我们使用 'coco',
    karpathy_json_path: 带有分类和标题的Karpathy JSON 文件路径
    image_folder: 包含下载图像的文件夹
    captions_per_image: 每张图片采样的标题数量
    min_word_freq: 出现频率低于此阈值的单词被划分到 <unk>s
    output_folder: 保存文件的文件夹
    max_len: 最大采样标题长度
    """
    assert dataset in {'coco'}

    # 读取 Karpathy JSON 文件
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # 读取每个图像的路径和标题
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # 更新词频
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # 合理性检测
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # 创建字图 word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # 为所有输出文件创建一个根名称
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # 将字图存放到 JSON 中
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # 采样每个图像的标题，将图像保存到 HDF5 文件，并将标题及其长度保存到 JSON 文件
    random.seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # 记录为每张图片采样的标题数量
            h.attrs['captions_per_image'] = captions_per_image

            # 在HDF5文件中创建数据集，用来存储图像
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # 采样标题
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [random.choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = random.sample(imcaps[i], k=captions_per_image)

                # 合理性检测
                assert len(captions) == captions_per_image

                # 读取图像
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = resize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # 将图像保存到HDF5文件中
                images[i] = img

                for j, c in enumerate(captions):
                    # 标题编码
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # 记录标题长度
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # 合理性检测
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # 将编码的标题及其长度保存到 JSON文件中
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)
def init_embedding(embeddings):
    """
    用均值填充embedding tensor.

    embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)