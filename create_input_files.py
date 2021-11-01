from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='/home3/jiachuang/course/nlp/data/caption_data/dataset_coco.json',
                       image_folder='/home3/jiachuang/course/nlp/data/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/home3/jiachuang/course/nlp/data/caption_data/',
                       max_len=50)