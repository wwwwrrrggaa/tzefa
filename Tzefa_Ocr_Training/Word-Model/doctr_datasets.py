from doctr.datasets import IMGUR5K
train_set = IMGUR5K(train=True, img_folder=r"E:/temp",
                    label_path=r"E:/temp")
img, target = train_set[0]
test_set = IMGUR5K(train=False, img_folder=r"E:/temp/IMGUR5K-Handwriting-Dataset/images",
                   label_path=r"E:/temp/IMGUR5K-Handwriting-Dataset/dataset_info/imgur5k_annotations.json")
img, target = test_set[0]