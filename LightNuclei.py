'''
This is the enterpoint for using LightNeuclei to predict on test data.
Use `LightNeuclei_test.py -h` to see descriptions of the arguments.
'''
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="LightNeuclei prediction",
        epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-w', '-weight', default='./weights/pretrained.h5', type=str,
        help='Weight of a pretrained model (example: ./weights/pretrained.h5)')
   
    parser.add_argument('-o', '--outputdir', default='./', type=str,
        help='Output directory')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # PCA_correct
    # input : stage1_train
    # output: train_image, train_image_flip
    os.system('python3 flip_train.py')
    print('flip_train.py completed')

    # split image and mask into patches
    # input: stage1_train
    # output: stage1_train_200_iflarge
    os.system('python3 split_train_and_mask_into_200_iflarge.py')
    print('split_train_and_mask_into_200_iflarge.py completed')

    # split image and mask files from the stage1_train_200_iflarge dir, PCA correct the images
    # input: stage1_train_200_iflarge
    # output: train_image_flip_iflarge_min200
    os.system('python3 flip_train_iflarge_min200.py')
    print('flip_train_iflarge_min200.py completed')

    # Produce split_label, i.e. one file annotating masks for each image
    # input: stage1_train_labels.csv
    # output: Split_label
    os.system('perl split_stage1_train_labels.pl')
    print('split_stage1_train_labels.pl completed')

    # generate train_label (.png) from split_label (.txt)
    # input: split_label
    # output: Train_label
    os.system('python3 generate_train_label.py')
    print('genereate_train_label.py completed')

    # Pad train_image into 1024
    # input: train_images
    # output: train_image_pad
    os.system('python3 pad_train_to_1024.py')
    print('pad_train_to_1024.py completed')

    # Pad train_label into 1024
    # input: train_label
    # output: train_label_pad
    os.system('python3 pad_train_label_to_1024.py')
    print('pad_train_label_to_1024.py completed')

    # generate list_test, list_train
    # with a random seed: e.g. 2018
    os.system('perl step1_split.pl 2018')
    print('step1_split.pl completed')

    

if __name__ == '__main__':
    main()

