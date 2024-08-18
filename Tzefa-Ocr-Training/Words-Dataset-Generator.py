import argparse
import hashlib
import json
import os
import random
import shutil
from itertools import chain
from multiprocessing import Pool, Manager, Process

import cv2
import numpy as np
import requests
import scipy
from PIL import Image, ImageDraw, ImageFont


def getfolderdict(path):
    # path=r'C:\DataSets\Words'
    worddict = {}
    sep = r'\\'
    for word in os.listdir(path):
        worddict[word] = path + word
    return worddict


def reorganizefonts():
    path = r'C:\Temp\fonts-main (1)\fonts-main\ofl'
    dest = r'C:\DataSets\fonts'
    sep = r'\\'
    for fontdir in os.listdir(path):
        for file in os.listdir(path + sep + fontdir):
            if file.endswith('ttf'):
                fontpath = path + sep + fontdir + sep + file
                try:
                    shutil.move(fontpath, dest)
                except(shutil.Error):
                    pass


def extractiamimages():
    wordspath = r'C:\Temp\archive\iam_words\words'
    worddict = getfolderdict()
    words = open(r"C:\Temp\archive\words_new.txt")
    wordsdest = r'C:\DataSets\Words'
    sep = '\\'
    words = words.readlines()[18:-1]
    words = [i[0:-1] for i in words]
    words = [i.split() for i in words]
    words = [[i[0].split(sep='-'), i[-1]] for i in words if i[-1].isalpha()]
    dict2 = {}
    for Word in words:
        if (Word[1] not in worddict and Word[1] not in dict2):
            dict2[Word[1]] = 0
            os.mkdir(wordsdest + sep + Word[1])
    for Word in words:
        word = Word[0]
        sourcepath = wordspath + sep + word[0] + sep + word[0] + '-' + word[1] + sep + word[0] + '-' + word[1] + '-' + \
                     word[2] + '-' + word[3] + '.png'
        shutil.move(sourcepath, wordsdest + sep + Word[1])


def CheckOne(f):
    try:
        im = Image.open(f)
        im.verify()
        im.close()
        # DEBUG: print(f"OK: {f}")
        return
    except (IOError, OSError, Image.DecompressionBombError):
        # DEBUG: print(f"Fail: {f}")
        return f


def checkbadimage(f):
    try:
        im = Image.open(f)
        im.verify()
        # DEBUG: print(f"OK: {f}")
        if (im == None):
            im.close()
            return f
    except (IOError, OSError, Image.DecompressionBombError, SyntaxError):
        # DEBUG: print(f"Fail: {f}")
        return f
    return None


def checkfolder():  # Create a pool of processes to check files
    p = Pool(process)

    # Create a list of files to process
    # files = [[worddir+'\\'+os.listdir(r'C:\DataSets\Words\\'+worddir)] for worddir in os.listdir(r'C:\DataSets\Words')]
    files = [worddir for worddir in os.listdir(r'C:\DataSets\Numbers')]
    files = [
        [r'C:\DataSets\Numbers' + '\\' + word + '\\' + file for file in os.listdir(r'C:\DataSets\Numbers\\' + word)] for
        word in files]
    files = list(chain.from_iterable(files))
    print(files)

    print(f"Files to be checked: {len(files)}")

    # Map the list of files to check onto the Pool
    result = p.map(checkbadimage, files)

    # Filter out None values representing files that are ok, leaving just corrupt ones
    result = list(filter(None, result))
    print(f"Num corrupt files: {len(result)}")
    print(result)
    for file in result:
        print(file)
    print('ahhhh')


def extractGNHK():
    folder = r'C:\Temp\train_data\train'
    dir = os.listdir(folder)
    addeddict = {}
    worddict = getfolderdict()
    for index in range(0, len(dir), 2):
        imagepath = folder + '\\' + dir[index]
        jsonpath = folder + '\\' + dir[index + 1]
        jsonfile = json.load(open(jsonpath))
        imagefile = Image.open(imagepath)
        for word in jsonfile:
            if (word['text'].isalpha()):
                cords = word['polygon']
                # print(word['text'],word['polygon'])
                minx = min(cords['x0'], cords['x1'], cords['x2'], cords['x3'])
                maxx = max(cords['x0'], cords['x1'], cords['x2'], cords['x3'])
                miny = min(cords['y0'], cords['y1'], cords['y2'], cords['y3'])
                maxy = max(cords['y0'], cords['y1'], cords['y2'], cords['y3'])
                image = imagefile.crop((minx, miny, maxx, maxy))
                # image.show()
                dest = r'C:\DataSets\Words' + '\\' + word['text']
                if (word['text'] in addeddict):
                    addeddict[word['text']] += 1
                    num = str(addeddict[word['text']])
                    name = 'GNHKcrop' + num + '.png'
                    path = dest + '\\' + name
                    image.save(path)
                elif (word['text'] in worddict):
                    addeddict[word['text']] = 1
                    num = str(addeddict[word['text']])
                    name = 'GNHKcrop' + num + '.png'
                    path = dest + '\\' + name
                    image.save(path)

                else:
                    addeddict[word['text']] = 1
                    num = str(addeddict[word['text']])
                    name = 'GNHKcrop' + num + '.png'
                    path = dest + '\\' + name
                    os.mkdir(dest)
                    image.save(path)

            else:
                pass


def extractsmalldataset():
    path = r'C:\Temp\archive\data\data\Capital'
    dest = r'C:\DataSets\Words'
    worddict = getfolderdict()
    for folder in os.listdir(path):
        worddest = dest + '\\' + folder
        if (folder not in worddict):
            os.mkdir(worddest)
        for file in os.listdir(path + '\\' + folder + '\\' + folder):
            shutil.move(path + '\\' + folder + '\\' + folder + '\\' + file, worddest)


def getfontlist():
    path = r'C:\DataSets\fonts'
    fontlist = ['0' for i in range(len(os.listdir(path)))]
    count = 0
    for file in os.listdir(path):
        filepath = path + '\\' + file
        fontlist[count] = filepath
        count += 1
    return fontlist


def generate_captcha(width, height, length):
    ### batch one
    fontlist = getfontlist()
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    captcha_text = ''.join(random.choice(characters) for _ in range(length))

    image = Image.new('RGB', (width, height),
                      color=(random.randint(128, 255), random.randint(128, 255), random.randint(128, 255)))

    draw = ImageDraw.Draw(image)
    # font = ImageFont.truetype(random.choice(fontlist), size=40)

    # Apply random rotation to each character
    for i, char in enumerate(captcha_text):
        font = ImageFont.truetype(random.choice(fontlist), size=random.randint(30, 60))
        char_image = Image.new('RGBA', (random.randint(32, 64), random.randint(32, 64)),
                               (random.randint(0, 240), random.randint(0, 240), random.randint(0, 240), 0))
        char_draw = ImageDraw.Draw(char_image)
        char_draw.text((0, 0), char, (random.randint(0, 128), random.randint(0, 128), random.randint(0, 128)),
                       font=font)
        char_image = char_image.rotate(random.randint(-45, 45), expand=1)

        # Apply random distortion
        distorted_image = Image.new('RGBA', char_image.size)
        for x in range(char_image.width):
            for y in range(char_image.height):
                src_x = int(x + 0)
                src_y = int(y + 0)
                if 0 <= src_x < char_image.width and 0 <= src_y < char_image.height:
                    distorted_image.putpixel((x, y), char_image.getpixel((src_x, src_y)))

        image.paste(distorted_image, (i * 64 + random.randint(-22, 22) + 10, random.randint(0, 60)), distorted_image)

    # image = image.filter(ImageFilter.GaussianBlur(radius=1))
    # image.show()
    return captcha_text, image


def createimagesfromfonts(worddir, addeddict, worddict):
    # worddir=r"C:\DataSets\Words3"
    sep = "\\"
    for i in range(4096 * 16):
        try:
            wordname, img = generate_captcha(1024, 128, random.randint(6, 20))
            if (wordname in addeddict):
                addeddict[wordname] += 1
                filepath = worddir + sep + wordname + sep + 'genbatch1-number' + str(addeddict[wordname]) + ".png"
                img.save(filepath)
            elif (wordname in worddict):
                addeddict[wordname] = 1
                filepath = worddir + sep + wordname + sep + 'genbatch1-number' + str(addeddict[wordname]) + ".png"
                img.save(filepath)
                pass

            else:
                addeddict[wordname] = 1
                filepath = worddir + sep + wordname + sep + 'genbatch1-number' + str(addeddict[wordname]) + ".png"
                os.mkdir(worddir + sep + wordname)
                img.save(filepath)
                pass
        except OSError:
            pass


def simplifyfolder(path):
    filepath = r"C:\DataSets\textfiles\train.txt"
    sep = '\\'
    textfile = open(filepath, 'w')
    for worddir in os.listdir(path):
        worddirpath = path + sep + worddir
        for file in os.listdir(worddirpath):
            textfile.write(worddirpath + sep + file + " " + worddir + '\n')
            # print(worddirpath+sep+file,worddir"C:\DataSets\Words2\train.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Processing imgur5K dataset download...")
    parser.add_argument(
        "--dataset_info_dir",
        type=str,
        default="dataset_info",
        required=False,
        help="Directory with dataset information",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="images",
        required=False,
        help="Directory path to download the image",
    )
    args = parser.parse_args()
    return args


# Image hash computed for image using md5..
def compute_image_hash(img_path):
    return hashlib.md5(open(img_path, 'rb').read()).hexdigest()


# Create a sub json based on split idx
def _create_split_json(anno_json, _split_idx):
    split_json = {}

    split_json['index_id'] = {}
    split_json['index_to_ann_map'] = {}
    split_json['ann_id'] = {}

    for _idx in _split_idx:
        # Check if the idx is not bad
        if _idx not in anno_json['index_id']:
            continue

        split_json['index_id'][_idx] = anno_json['index_id'][_idx]
        split_json['index_to_ann_map'][_idx] = anno_json['index_to_ann_map'][_idx]

        for ann_id in split_json['index_to_ann_map'][_idx]:
            split_json['ann_id'][ann_id] = anno_json['ann_id'][ann_id]

    return split_json


def get5kimgs():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = r"C:\DataSets\Words2"
    # Create a hash dictionary with image index and its correspond gt hash
    with open(
            r"C:\Temp\facebookresearch IMGUR5K-Handwriting-Dataset 756a9ac9ed5201345661e1d9b7a5eb53502b97d5 dataset_info" + "/imgur5k_hashes.lst",
            "r", encoding="utf-8") as _H:
        hashes = _H.readlines()
        hash_dict = {}

        for hash in hashes:
            hash_dict[f"{hash.split()[0]}"] = f"{hash.split()[1]}"

    tot_evals = 0
    num_match = 0
    invalid_urls = []
    # Download the urls and save only the ones with valid hash o ensure underlying image has not changed
    for index in list(hash_dict.keys()):
        image_url = f'https://i.imgur.com/{index}.jpeg'
        print(image_url)
        img_data = requests.get(image_url).content
        # if len(img_data) < 100:
        # print(f"URL retrieval for {index} failed!!\n")
        # invalid_urls.append(image_url)
        #   continue
        print(img_data)
        with open(f'{args.output_dir}/{index}.jpg', 'wb') as handler:
            handler.write(img_data)

        compute_image_hash(f'{args.output_dir}/{index}.jpg')
        tot_evals += 1
        # if hash_dict[index] != compute_image_hash(f'{args.output_dir}/{index}.jpg'):
        # print(f"For IMG: {index}, ref hash: {hash_dict[index]} != cur hash: {compute_image_hash(f'{args.output_dir}/{index}.jpg')}")
        # os.remove(f'{args.output_dir}/{index}.jpg')
        # invalid_urls.append(image_url)
        # continue
        # else:
        # num_match += 1

    # Generate the final annotations file
    # Format: { "index_id" : {indexes}, "index_to_annotation_map" : { annotations ids for an index}, "annotation_id": { each annotation's info } }
    # Bounding boxes with '.' mean the annotations were not done for various reasons

    _F = np.loadtxt(f'{args.dataset_info_dir}/imgur5k_data.lst', delimiter="\t", dtype=np.str, encoding="utf-8")
    anno_json = {}

    anno_json['index_id'] = {}
    anno_json['index_to_ann_map'] = {}
    anno_json['ann_id'] = {}

    cur_index = ''
    for cnt, image_url in enumerate(_F[:, 0]):
        if image_url in invalid_urls:
            continue

        index = image_url.split('/')[-1][:-4]
        if index != cur_index:
            anno_json['index_id'][index] = {'image_url': image_url, 'image_path': f'{args.output_dir}/{index}.jpg',
                                            'image_hash': hash_dict[index]}
            anno_json['index_to_ann_map'][index] = []

        ann_id = f"{index}_{len(anno_json['index_to_ann_map'][index])}"
        anno_json['index_to_ann_map'][index].append(ann_id)
        anno_json['ann_id'][ann_id] = {'word': _F[cnt, 2], 'bounding_box': _F[cnt, 1]}

        cur_index = index

    json.dump(anno_json, open(f'{args.dataset_info_dir}/imgur5k_annotations.json', 'w'), indent=4)

    # Now split the annotations json in train, validation and test jsons
    splits = ['train', 'val', 'test']
    for split in splits:
        _split_idx = np.loadtxt(f'{args.dataset_info_dir}/{split}_index_ids.lst', delimiter="\n", dtype=np.str)
        split_json = _create_split_json(anno_json, _split_idx)
        json.dump(split_json, open(f'{args.dataset_info_dir}/imgur5k_annotations_{split}.json', 'w'), indent=4)

    print(f"MATCHES: {num_match}/{tot_evals}\n")


def fixdir(path):
    for worddir in os.listdir(path):
        for file in os.listdir(path + '\\' + worddir):
            with Image.open(path + '\\' + worddir + '\\' + file) as img:
                width, height = img.size
            if (width * height < 100):
                os.remove(path + '\\' + worddir + '\\' + file)
    for worddir in os.listdir(path):
        diri = os.listdir(path + '\\' + worddir)
        if (len(diri) == 0):
            print(worddir)
            os.rmdir(path + '\\' + worddir)


def getoptimalcrop(crops):
    cropminx = min([i[0] for i in crops])
    cropminy = min([i[1] for i in crops])
    cropmaxx = max([i[0] + i[2] for i in crops])
    cropmaxy = max([i[0] + i[3] for i in crops])
    return [cropminx, cropminy, cropmaxx, cropmaxy]


def downloadsmall5k(path):
    mat = scipy.io.loadmat(r"C:\Temp\IIIT5K-Word_V3.0\IIIT5K\testCharBound.mat")
    dataset = mat["testCharBound"]
    imagedir = r"C:\Temp\IIIT5K-Word_V3.0\IIIT5K"
    sep = "\\"
    imagefilename = "A"
    worddict = getfolderdict(path)
    addeddict = {}
    worddir = path
    for batch in dataset:
        for imagedata in batch:
            wordname = imagedata[1][0]
            filename = imagedata[0][0].replace('/', sep)
            imagefilename = imagedir + sep + filename
            img = Image.open(imagefilename)
            if (wordname.isalpha()):
                if (wordname in addeddict):
                    addeddict[wordname] += 1
                    filepath = worddir + sep + wordname + sep + 'genbatch2-number' + str(addeddict[wordname]) + ".png"
                    img.save(filepath)
                elif (wordname in worddict):
                    addeddict[wordname] = 1
                    filepath = worddir + sep + wordname + sep + 'genbatch1-number' + str(addeddict[wordname]) + ".png"
                    img.save(filepath)

                else:
                    addeddict[wordname] = 1
                    filepath = worddir + sep + wordname + sep + 'genbatch1-number' + str(addeddict[wordname]) + ".png"
                    os.mkdir(worddir + sep + wordname)
                    img.save(filepath)


def insert_image_to_folder(image_value: str, image: cv2.Mat, save_folder_path: str, exisiting_words: dict,
                           added_words: dict):
    if (image_value in added_words):
        added_words[image_value] += 1
        filepath = save_folder_path + image_value + "\\" + "Imgur_5k_" + str(added_words[image_value]) + ".jpg"

    elif (image_value in exisiting_words):
        added_words[image_value] = 1
        filepath = save_folder_path + image_value + "\\" + "Imgur_5k_" + str(added_words[image_value]) + ".jpg"

    else:
        added_words[image_value] = 1
        os.mkdir(save_folder_path + image_value)
        filepath = save_folder_path + image_value + "\\" + "Imgur_5k_" + str(added_words[image_value]) + ".jpg"
    cv2.imwrite(filepath, image)


def extract_cropped_image_from_Text_Ocr(dataset_path, line, save_folder_path):
    global existing_words, added_words
    if (not (line[2].isnumeric())):
        return None
    else:
        img_file_path = dataset_path + line[0].replace('train/', r'\\')
        image = cv2.imread(img_file_path)
        bbp = [int(i) for i in line[1]]
        cropped_image = image[bbp[1]:bbp[1] + bbp[3], bbp[0]:bbp[0] + bbp[2]]

        insert_image_to_folder(line[2], cropped_image, save_folder_path, existing_words, added_words)
        return img_file_path


def extract_Text_ocr_dataset(dataset_path, json_path, save_folder_path):
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)
    image_names = {img: [json_data["imgs"][img]["file_name"], [], "."] for img in json_data["imgs"]}
    # print(image_names)
    ###add words and bboxes
    for data in json_data["anns"]:
        image_names[data.split('_')[0]][1] = json_data["anns"][data]['bbox']  ### extracts id and puts data in
        image_names[data.split('_')[0]][2] = json_data["anns"][data]['utf8_string']
    for line in image_names:
        extract_cropped_image_from_Text_Ocr(dataset_path, image_names[line], save_folder_path)
    ### add


def extract_Imgur5K_dataset(dataset_path, save_folder_path,existing_words):
    import tensorflow as tf

    img_shape_config = (32, 128, 1)
    added_words={}
    dataset_path
    for file in os.listdir(dataset_path):
        tfr_path = dataset_path+file
        dataset = tf.data.TFRecordDataset(tfr_path)
        counter=0
        length=10000000
        #info_array = [{'image':"",'label':"", 'height':"", 'width':"",'chans':""} for i in range(length)]

        for raw_record in dataset.take(length):
            if(raw_record==None):
                break

            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            info={'image':"",'label':"", 'height':"", 'width':"",'chans':""}
            for k, v in example.features.feature.items():
                if k == 'image':
                    info[k] = v.bytes_list.value[0]
                elif k in ['label']:
                    #print(v.bytes_list.value[0])
                    info[k] = v.bytes_list.value[0]
                elif k in ['height', 'width','chans']:
                    #print(v.int64_list.value[0])
                    info[k] = v.int64_list.value[0]
            if(info['label'].isalpha()):
                image = tf.image.decode_png(info["image"], channels=info['chans'])
                image = tf.cast(image, tf.float32)
                image = tf.reshape(image, (info['height'], info['width'], info['chans']))
                image = tf.image.convert_image_dtype(image, tf.float32)
                #image = tf.image.encode_jpeg(image, quality=90)
                image=image.numpy()
                insert_image_to_folder(str(info['label'].decode('ascii')),image,save_folder_path,existing_words,added_words)
                #img_arr = np.frombuffer(info['image'], dtype=np.uint8).reshape(
                    #info['height'], info['width'])
                info={}
            counter+=1


def multiprocess():
    path = r"C:\DataSets\Words\\"
    p = Pool(processes=12)
    manger = Manager()
    addeddict = manger.dict()
    worddict = manger.dict()
    worddictlocal = getfolderdict(path)
    for key in worddictlocal:
        worddict[key] = worddictlocal[key]
    processlist = [0 for i in range(12)]
    for i in range(12):
        processlist[i] = Process(target=createimagesfromfonts, args=[path, addeddict, worddict])
        processlist[i].start()
    for i in range(12):
        processlist[i].join()


if __name__ == '__main__':
    save_folder_path = r"C:\DataSets\Words\\"
    existing_words = getfolderdict(save_folder_path)
    added_words = {}
    extract_Imgur5K_dataset(r"C:\Users\yonat\Downloads\archive\IMGUR5K_val\\",save_folder_path,existing_words)

    #simplifyfolder(save_folder_path)
    # extract_Text_ocr_dataset(r"C:\Users\yonat\Downloads\train_val_images\train_images",r"C:\Users\yonat\Downloads\TextOCR_0.1_train.json",save_folder_path)

    # simplifyfolder(path)
