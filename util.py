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

def load_embeddings(emb_file, word_map):
    """
    为指定的词图创建embedding tensor，以加载到模型之中。
    emb_file: 包含嵌入的文件（以 GloVe 格式存储）
    word_map: 字图
    return: 返回embeddings维度，顺序和字图相同
    """

    # 计算 embedding 维度
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # 创造tensor来保存embeddings, 初始化
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # 读取嵌入文件
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # 忽略不在训练集中的单词
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim
def clip_gradient(optimizer, grad_clip):
    """
    在反向传播期间负责避免梯度爆炸。
    optimizer: 可以剪裁梯度的优化器
    grad_clip: 夹值
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    保存模型的checkpoint.
    data_name: 已处理数据集的名称
    epoch: epoch number
    epochs_since_improvement: 自上次改进 BLEU-4 分数以来的 epoch number
    encoder: 编码模型
    decoder: 解码模型
    encoder_optimizer: 如果需要微调，更新编码器权重的优化器
    decoder_optimizer: 更新解码器权重的优化器
    bleu4: 验证该时期的 BLEU-4 分数
    is_best: 是否是迄今为止最好的checkpoint?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # 如果这个检查点是目前最好的，则保存
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    实时更新目标的总和、均值、计数值。
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    按指定的系数缩小学习率。

    optimizer: 必须缩小学习率的优化器.
    shrink_factor: 从 (0, 1)中取恰当缩小因子，再乘以学习率
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    根据预测标签和真实标签计算 top-k 准确度。

    scores: 模型分数
    targets: 真实分数
    k: K的前k个的精度
    返回： top-k的准确率
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)