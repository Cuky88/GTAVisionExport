# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
from os import listdir, makedirs
from os.path import isfile, join
import json
import argparse
import numpy as np
import shutil
import glob

def gta5_read(path, file):
    with open(path+file, 'r') as reader:
        for line in reader:
            #print(line)
            yield line

def preProcess():

    dst = 'final'
    try:
        makedirs(dst) # create destination directory, if needed (similar to mkdir -p)
    except OSError:
        # The directory already existed, nothing to do
        pass

    annodir = 'Data/'
    files = [f for f in listdir(annodir) if isfile(join(annodir, f))]

    for c, file in enumerate(files):
        if file.split('.')[-1] == 'txt':
            label = gta5_read(annodir, file)

            # Filename: bb_4_35.txt
            num_frame = int(file.split('_')[1])

            for s in label:
                with open("final/bb_" + str(num_frame) + ".txt", "a") as myfile:
                    myfile.write(s)

def createJson(phase, off):
    preProcess()

    annodir = 'final/'
    files = [f for f in listdir(annodir) if isfile(join(annodir, f))]

    i = 0
    j = 0
    total = []

    yos = 0
    xos = 0

    for c, file in enumerate(files):
        if file.split('.')[-1] == 'txt':
            rec = []
            rec2 = []
            label = gta5_read(annodir, file)
            num_frame = int(file.split('_')[1][:-4])
            #frame = "image_" + str(num_frame) + '.jpeg'

            if not off:
                frame = "cross_" + str(num_frame) + '.jpeg'
            else:
                frame = "cross_" + str(num_frame+1) + '.jpeg'

            for s in label:
                if '%' not in s:
                    coords = s.split(' ')
                    #print('frame %s: x1 %s, y1 %s - x2 %s, y2 %s'%(frame, x1, y1, x2, y2))
                    i += 1

                    try:
                        rec.append({"x1": int(coords[0]), "y1": int(coords[1]), 
                                    "x2": int(coords[2]), "y2": int(coords[3])})

                        rec2.append({"x1": int(coords[4]) + xos, "y1": int(coords[5]) + yos, 
                                        "x2": int(coords[6]) + xos, "y2": int(coords[7]) + yos,
                                        "x3": int(coords[8]) + xos, "y3": int(coords[9]) + yos, 
                                        "x4": int(coords[10]) + xos, "y4": int(coords[11]) + yos,
                                        "x5": int(coords[12]) + xos, "y5": int(coords[13]) + yos, 
                                        "x6": int(coords[14]) + xos, "y6": int(coords[15]) + yos,
                                        "x7": int(coords[16]) + xos, "y7": int(coords[17]) + yos, 
                                        "x8": int(coords[18]) + xos, "y8": int(coords[19]) + yos})
                    except:
                        print("cross_" + str(num_frame) + '.jpeg')
                        continue

            total.append({'frame': num_frame, 'rects': rec, 'rects_3d': rec2, 'image_path': (annodir + frame)})
            j += 1

    #print(i)
    #print(j)

    with open( phase + '_boxes.json', 'w') as outfile:
        json.dump(total, outfile, indent=2)
        outfile.write('\n')

    for jpeg_file in glob.iglob('Data/*.jpeg'):
        shutil.copy2(jpeg_file, annodir)

def createVal():
    # Randomly select ~1000 images from train set and make ist the val set
    valset = np.random.randint(19060, size=(1, 1000))
    valset = set(valset[0])
    print(len(valset))

    train = []
    train_new = []
    val = []
    filenames = []

    with open('../data/GTA5/train_boxes.json', 'r') as reader:
        data = json.load(reader)
        for f in data:
            train.append(f)

    flag = 0
    for i, file in enumerate(train):
        flag = 0
        for ind in valset:
            if i == ind:
                flag = 1
                val.append(file)
                filenames.append(file['image_path'].strip('/')[1])
        if not flag:
            train_new.append(file)

    # annodir_train = 'data/COD20K/annotations/train/'
    # files = [f for f in listdir(annodir_train) if isfile(join(annodir_train, f))]
    #
    # try:
    #     for name in filenames:
    #         for file in files:
    #             if name == file:
    #                 shutil.copy(annodir_train + file, 'data/COD20K/annotations/val')
    # except IOError as e:
    #     print("Unable to copy file. %s" % e)
    # except:
    #     print("Unexpected error!")

    with open('../data/GTA5/val_boxes.json', 'w') as outfile:
        json.dump(val, outfile, indent=2)
        outfile.write('\n')
    print(len(val))

    with open('../data/GTA5/train_new_boxes.json', 'w') as writer:
        json.dump(train_new, writer, indent=2)
        writer.write('\n')
    print(len(train_new))

def showBB(img, rects, name):
    for bb in rects:
        print(bb['x1'], bb['y1']), (bb['x2'], bb['y2'])
        img = cv2.rectangle(img, (bb['x1'], bb['y1']), (bb['x2'], bb['y2']), (0, 0, 255), 1)

    plt.imshow(img)
    plt.show()

    #cv2.imshow(name,img)

    #key = cv2.waitKey(0)
    #cv2.destroyAllWindows()


def show3DBB(img, rects, name):
    '''
    x1, y1 = FUL
    x2, y2 = FLL
    x3, y3 = FLR
    x4, y4 = FUR
    x5, y5 = BLR
    x6, y6 = BUR
    x7, y7 = BUL
    x8, y8 = BLL
    '''
    for bb in rects:
        print (bb['x1'], bb['y1']), (bb['x2'], bb['y2']), (bb['x3'], bb['y3']), (bb['x4'], bb['y4']), 
        (bb['x5'], bb['y5']), (bb['x6'], bb['y6']), (bb['x7'], bb['y7']), (bb['x8'], bb['y8']) 

        img = cv2.line(img, (bb['x1'], bb['y1']), (bb['x2'], bb['y2']), (255, 0, 0), 1)
        img = cv2.line(img, (bb['x2'], bb['y2']), (bb['x3'], bb['y3']), (255, 0, 0), 1)
        img = cv2.line(img, (bb['x3'], bb['y3']), (bb['x4'], bb['y4']), (255, 0, 0), 1)
        img = cv2.line(img, (bb['x4'], bb['y4']), (bb['x1'], bb['y1']), (255, 0, 0), 1)

        img = cv2.line(img, (bb['x5'], bb['y5']), (bb['x6'], bb['y6']), (255, 0, 0), 1)
        img = cv2.line(img, (bb['x6'], bb['y6']), (bb['x7'], bb['y7']), (255, 0, 0), 1)
        img = cv2.line(img, (bb['x7'], bb['y7']), (bb['x8'], bb['y8']), (255, 0, 0), 1)
        img = cv2.line(img, (bb['x8'], bb['y8']), (bb['x5'], bb['y5']), (255, 0, 0), 1)

        img = cv2.line(img, (bb['x6'], bb['y6']), (bb['x4'], bb['y4']), (255, 0, 0), 1)
        img = cv2.line(img, (bb['x7'], bb['y7']), (bb['x1'], bb['y1']), (255, 0, 0), 1)
        img = cv2.line(img, (bb['x8'], bb['y8']), (bb['x2'], bb['y2']), (255, 0, 0), 1)
        img = cv2.line(img, (bb['x5'], bb['y5']), (bb['x3'], bb['y3']), (255, 0, 0), 1)

    plt.imshow(img)
    plt.show()

    #cv2.imshow(name, img)

def checkAnno(check):
    anno = []
    with open(check, 'r') as reader:
        data = json.load(reader)
        for f in data:
            anno.append(f)

    for f in anno:
        for bb in f['rects']:
            if bb['y1'] >= bb['y2']:
                print('Error y1 greater-equal y2 in %s'%f['image_path'])

def main():
    print("Starting")
    
    plt.figure(figsize=(20,18))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default=None, type=str)
    parser.add_argument('--show', default=None, type=int)
    parser.add_argument('--check', default=None, type=str)
    parser.add_argument('--td', default=None, type=int)
    parser.add_argument('--slide2', default=None, type=int)
    parser.add_argument('--slide3', default=None, type=int)
    parser.add_argument('--off', default=1, type=int)
    args = parser.parse_args()

    if args.phase is not None:
        print("Calling createJson")
        createJson(args.phase,args.off)
    elif args.phase == 'val':
        createVal()
    elif args.check is not None:
        checkAnno(args.check)

    if args.show is not None:
        with open('train_boxes.json', 'r') as reader:
            data = json.load(reader)
            for img in data:
                if img['frame'] == args.show:
                    print(img['image_path'])
                    img_loaded = cv2.cvtColor(cv2.imread(img['image_path'], 1), cv2.COLOR_BGR2RGB)
                    print(img['image_path'])

                    showBB(img_loaded, img['rects'], img['image_path'])

    if args.td is not None:
        with open('train_boxes.json', 'r') as reader:
            data = json.load(reader)
            for img in data:
                if img['frame'] == args.td:
                    img_loaded = cv2.cvtColor(cv2.imread(img['image_path'], -1), cv2.COLOR_BGR2RGB)
                    print(img['image_path'])

                    show3DBB(img_loaded, img['rects_3d'], img['image_path'])

    if args.slide2 is not None:
        with open('train_boxes.json', 'r') as reader:
            data = json.load(reader)
            for img in data:
                img_loaded = cv2.cvtColor(cv2.imread(img['image_path'], -1), cv2.COLOR_BGR2RGB)
                print(img['image_path'])

                showBB(img_loaded, img['rects_3d'], img['image_path'])

                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if key == 27: # escape
                    break

            # close any open windows
            cv2.destroyAllWindows()

    if args.slide3 is not None:
        with open('train_boxes.json', 'r') as reader:
            data = json.load(reader)
            for img in data:
                img_loaded = cv2.cvtColor(cv2.imread(img['image_path'], -1), cv2.COLOR_BGR2RGB)
                print(img['image_path'])

                show3DBB(img_loaded, img['rects_3d'], img['image_path'])

                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if key == 27: # escape
                    break

            # close any open windows
            cv2.destroyAllWindows()



if __name__ == '__main__':
    main()