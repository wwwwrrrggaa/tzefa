import os,random
import shutil
import sys
from PIL import Image,ImageDraw,ImageFont
from multiprocessing import Pool,Manager,Process
import numpy as np
import scipy.io
import csv
import numpy as np
import h5py
import mat73

def addpicturetodataset(wordname,filename,worddir,folderdict,addeddict):
    sep=r"\\"
    img=Image.open(filename)
    if(wordname in addeddict):
        addeddict[wordname] += 1
        filepath = worddir + sep + wordname + sep + 'batch2-number' + str(addeddict[wordname]) + ".png"
        img.save(filepath)
    elif(wordname in folderdict):
        addeddict[wordname] = 1
        filepath = worddir + sep + wordname + sep + 'batch1-number' + str(addeddict[wordname]) + ".png"
        img.save(filepath)
    else:
        addeddict[wordname] = 1
        filepath = worddir + sep + wordname + sep + 'batch1-number' + str(addeddict[wordname]) + ".png"
        os.mkdir(worddir + sep + wordname)
        img.save(filepath)
def getfolderdict(path):
    #path=r'C:\DataSets\Words'
    worddict={}
    sep=r'\\'
    for word in os.listdir(path):
        worddict[word]=path+word
    return worddict
def getfontlist():
    path=r'C:\DataSets\fonts'
    fontlist=['0' for i in range(len(os.listdir(path)))]
    count=0
    for file in os.listdir(path):
        filepath=path+'\\'+file
        fontlist[count]=filepath
        count+=1
    return fontlist
def generate_captcha(width, height, length):
    ### batch one
    fontlist=getfontlist()
    characters = "1234567890"
    captcha_text = ''.join(random.choice(characters) for _ in range(length))

    image = Image.new('RGB', (width, height), color=(random.randint(128,255), random.randint(128,255), random.randint(128,255)))
    for i, char in enumerate(captcha_text):
        font = ImageFont.truetype(random.choice(fontlist), size=random.randint(30, 60))
        char_image = Image.new('RGBA', (random.randint(int(width/(length*2)), int(width/(length))), random.randint(int(height/(10)), int(height/(2)))),
                               (random.randint(0, 240), random.randint(0, 240), random.randint(0, 240), 0))
        char_draw = ImageDraw.Draw(char_image)
        mixer=random.randint(0,1)
        char_draw.text((0, 0), char, (random.randint(0, 128+128*(abs(1-mixer))), random.randint(0, 128+128*(abs(-1-mixer))), random.randint(0, 256*mixer+128)),
                       font=font)
        char_image = char_image.rotate(random.randint(-45, 45), expand=1)

        # Apply random distortion
        distorted_image = Image.new('RGBA', char_image.size)
        for x in range(char_image.width):
            for y in range(char_image.height):
                src_x = int(x + 0.5)
                src_y = int(y + 0.5)
                if 0 <= src_x < char_image.width and 0 <= src_y < char_image.height:
                    distorted_image.putpixel((x, y), char_image.getpixel((src_x, src_y)))
        image.paste(distorted_image, (i*int(width/length)+random.randint(0,int(width/(length*2))), random.randint(a=0, b=abs(height-char_image.height))), distorted_image)
    return captcha_text,image
def createimagesfromfonts(worddir,addeddict,worddict):
    #worddir=r"C:\DataSets\Words3"
    sep="\\"
    for i in range(32):
        try:
            wordname,img=generate_captcha(384,384,random.randint(1,20))
            if (wordname in addeddict):
                addeddict[wordname]+=1
                filepath=worddir+sep+wordname+sep+'batch1-number'+str(addeddict[wordname])+".png"
                img.save(filepath)
            elif (wordname in worddict):
                addeddict[wordname] += 1
                filepath = worddir + sep + wordname + sep + 'batch1-number' + str(addeddict[wordname]) + ".png"
                img.save(filepath)
                pass

            else:
                addeddict[wordname] = 1
                filepath = worddir + sep + wordname + sep + 'batch1-number' + str(addeddict[wordname])+".png"
                os.mkdir(worddir+sep+wordname)
                img.save(filepath)
                pass
        except OSError:
            pass
def multiprocessing():
    path = r"C:\DataSets\Numbers"
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
def simplifyfolder(path):
    filepath=r"C:\DataSets\textfiles\numbertrain.txt"
    sep='\\'
    textfile=open(filepath,'w')
    for worddir in os.listdir(path):
        worddirpath=path+sep+worddir
        for file in os.listdir(worddirpath):
            textfile.write(worddirpath+sep+file+" "+worddir+'\n')
def unpack2500k(folder,path):
    for digit in os.listdir(folder):
        count=0
        for image in os.listdir(folder+r"\\"+digit):
            count+=1
            print(folder+r"\\"+digit+r"\\"+image,path+r"\\"+digit)
            shutil.move(folder+r"\\"+digit+r'\\'+image,path+r"\\"+digit)
def fixdir(path):
    for worddir in os.listdir(path):
        for file in os.listdir(path+'\\'+worddir):
            with Image.open(path+'\\'+worddir+'\\'+file) as img:
                width, height = img.size
            if(width*height<100):
                os.remove(path+'\\'+worddir+'\\'+file)
    for worddir in os.listdir(path):
        diri=os.listdir(path + '\\' + worddir)
        if(len(diri)==0):
            print(worddir)
            os.rmdir(path+'\\'+worddir)
def checkfolder():    # Create a pool of processes to check files
    p = Pool(processes=12)

    # Create a list of files to process
    #files = [[worddir+'\\'+os.listdir(r'C:\DataSets\Numbers\\'+worddir)] for worddir in os.listdir(r'C:\DataSets\Numbers')]
    files=[worddir for worddir in os.listdir(r'C:\DataSets\Numbers')]
    files=[[r'C:\DataSets\Numbers'+'\\'+word +'\\'+file for file in os.listdir(r'C:\DataSets\Numbers\\'+word)] for word in files]
    files=list(chain.from_iterable(files))
    print(files)

    print(f"Files to be checked: {len(files)}")

    # Map the list of files to check onto the Pool
    result = p.map(checkbadimage, files)

    # Filter out None values representing files that are ok, leaving just corrupt ones
    result = list(filter(None, result))
    print(f"Num corrupt files: {len(result)}")
    for file in result:
        os.remove(file)
def readmat2(file):
    mat = scipy.io.loadmat(file)
    print(mat)
def dumbmetric(stri):
    stri=stri.split(r"\\")
    stri=stri[-1][0:-4]
    return int(stri)
def addhndataset(folder,path):
    sep=r"\\"
    listi=os.listdir(folder)[-1]
    listi=folder+sep+listi
    valuelist=[]
    sum=""
    with open(listi) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            for value in row:
                for j in value:
                    if(j!=","):
                        sum+=j
                    else:
                        valuelist.append(sum)
                        sum = ""
    count=0
    sortedfiles=sorted(os.listdir(folder)[0:-1],key=dumbmetric)
    folderdict=getfolderdict(path)
    newdict={}
    for file in sortedfiles:
        print(valuelist[count],folder+sep+file)
        addpicturetodataset(valuelist[count],folder+sep+file,path,folderdict,newdict)
        count+=1
        pass
def unpackmnist(folder,path):
    pass
def unpacksvhn(file,folder,path):
    with h5py.File(file, 'r') as f:
        f.




if __name__ == '__main__':
    path=r'C:\DataSets\Numbers'
    #extractmnist("C:\Temp\Handwritten Numbers (HN)\Handwritten Numbers (HN)\handwritten_numbers_v1",path)
    unpacksvhn("C:\Temp\extra\extra\digitStruct.mat","C:\Temp\extra\extra",path)
