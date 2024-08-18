import math
from trdg.generators import GeneratorFromStrings

import multiprocessing
import random
import os

def getfolderdict(path):
    # path=r'C:\DataSets\Words'
    worddict = {}
    sep = r'\\'
    for word in os.listdir(path):
        worddict[word] = path + word
    return worddict


def generate_images_from_trdg(count, word_dir,addeddict, worddict):
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    words_list = [''.join(random.choice(characters) for _ in range(random.randint(1, 20))) for i in range(count)]  #
    generator = GeneratorFromStrings(strings=words_list, count=count,random_blur=True,blur=1,size=random.randint(56,128),random_skew=True,skewing_angle=5,distorsion_type=3,distorsion_orientation=2,width=random.randint(00, 1000),alignment=random.randint(0,2),text_color="#000000,#FFFFFF",is_handwritten=False)
    sep='\\'

    for img_tuple in generator:
        img=img_tuple[0]
        wordname=img_tuple[1]
        if img==None:
            pass
        elif (wordname in addeddict):
            addeddict[wordname] += 1
            filepath = word_dir + sep + wordname + sep + 'genbatch3-number' + str(addeddict[wordname]) + ".jpg"
            img.save(filepath)
        elif (wordname in worddict):
            addeddict[wordname] = 1
            filepath = word_dir + sep + wordname + sep + 'genbatch3-number' + str(addeddict[wordname]) + ".jpg"
            img.save(filepath)
            pass

        else:
            addeddict[wordname] = 1
            filepath = word_dir + sep + wordname + sep + 'genbatch3-number' + str(addeddict[wordname]) + ".jpg"
            os.mkdir(word_dir + sep + wordname)
            img.save(filepath)
            pass



def main():
    path="C:\\datasets\\Words"  # Replace with your dataset directory
    process_count = multiprocessing.cpu_count()  # Adjust the number of processes based on your system's resources'
    print(f"Number of processes: {process_count}")
    # =multiprocessing.Pool(process_count)
    processlist = [multiprocessing.Process() for i in range(process_count)]
    img_count = 1000*2
    batch_size = 1000
    batches = math.ceil(img_count / batch_size)
    data_dir = path  # Replace with your desired output directory

    manger = multiprocessing.Manager()
    addeddict = manger.dict()
    worddict = manger.dict()
    worddictlocal = getfolderdict(path)
    for key in worddictlocal:
        worddict[key] = worddictlocal[key]

    for i in range(batches):
        for i in range(process_count):
            processlist[i] = multiprocessing.Process(target=generate_images_from_trdg, args=[batch_size, data_dir,addeddict,worddict])
            processlist[i].start()
        for i in range(process_count):
            processlist[i].join()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
