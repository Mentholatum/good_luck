from main import caption_image_beam_search
from main import visualize_att
import torch
import json
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def caption(img='/home3/jiachuang/course/nlp/projects/good_luck/upload_files/20211210142029.png'):
    """
    function to generate imgage description
    :param img: path to image
    :return: description
    """

    parser = argparse.ArgumentParser(description='NLP course - Image Caption Generator - Good luck')
    parser.add_argument('--model', '-m', default='BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar',help='path to model')
    parser.add_argument('--word_map', '-wm', default='/home3/jiachuang/course/nlp/data/caption_data'
                                                     '/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json',
                        help='path to word map JSON')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    args = parser.parse_args()


    # Load model
    checkpoint = torch.load(args.model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, img, word_map, 5)
    alphas = torch.FloatTensor(alphas)

    # Visualize caption and attention of best sequence
    des = visualize_att(img, seq, alphas, rev_word_map, args.smooth)
    return des
if __name__ == '__main__':
    caption()
