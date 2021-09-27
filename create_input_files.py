from util import create_input_files

if __name__ == '__main__':
    #创建输入文件以及字图
    create_input_files(dataset='coco',
                       karpathy_json_path='/home3/jiachuang/course/nlp/data/caption data/dataset_coco.json',
                       image_folder='/home3/jiachuang/course/nlp/data/train2017/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/home3/jiachuang/course/nlp/data/caption data/',
                       max_len=50)