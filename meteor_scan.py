# -*- coding: utf-8 -*-

'''
Meteor Scan 
Author: Hu, Ying-Hao (hyinghao@hotmail.com)
Version: 2.1.1
Last modification date: 2022-06-04
Copyright 2022 Hu, Ying-Hao
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Description:
    The new approach basically leverage the semantic segmentation and
    line segment detection to find out the meteors

    line segment detection is used to figure out the meteors in the sky
    The algorithm is based on opencv.

    semantic segmentation is used to figure out the sky area,
    The algorithm is based on pixellib.
    As there're noises on the ground frequently. So basically, a lower bound
    is set up to filter out the detection of line segments
    
    Finally, use spatial cluster to merge all continuous frames
    cut them from the original video

    This approach is faster, however, it cannot cover 100% for all cases.
    video quality is crucial to this problem, high fidelity is must

You'd better run it under Linux or MacOS(ubuntu is preferred)
Windows is not supported, as the command /w os.system is written in a UNIX style
Anyway, you can change them
Directory structure:
meteor_scan
    ├── meteor01.mp4
    ├── meteor_scan.py
    ├── dump
    │   └── ...
    ├── output
    │   └── ...
    └── scanned
        └── ...
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import glob
import math
import numpy as np
import sys
import pixellib

from PIL import Image, ImageDraw, ImageEnhance
from sklearn.cluster import DBSCAN
from concurrent.futures import ProcessPoolExecutor
from pixellib.semantic import semantic_segmentation


def generate_mask(filename):
    print("find sky background area")

    si = semantic_segmentation()
    si.load_ade20k_model('deeplabv3_xception65_ade20k.h5')

    # define filenames
    maskFilename = filename[:-4]+"_mask.png"
    maskPasteFilename = filename[:-4]+"_maskPaste.png"
    brightenFilename = filename[:-4]+"_brighten.png"

    # generate a brightened image
    im = Image.open(filename)
    enhancer = ImageEnhance.Brightness(im)
    factor = 3 #gives original image
    im_output = enhancer.enhance(factor)
    im_output.save(brightenFilename)

    size = im.size

    # using semantic segmentation
    a, res, output = si.segmentAsAde20k(
        brightenFilename, 
        extract_segmented_objects=True, 
        output_image_name=maskFilename
    )
    os.system("rm %s" % brightenFilename)
    lowerBound = size[1]

    # try to figure out category of the largest area as a candidate for sky
    # final mask = mask of largest area | mask of sky
    area = []
    skyId = -1
    for i in range(len(res)):
        area.append(np.sum(res[i]["masks"].astype(np.uint)))
        if res[i]['class_name'] == 'sky':
            skyId = i
    mxId = np.argmax(np.array(area))
    if (skyId != -1):
        m = res[skyId]["masks"]
        if mxId != skyId:
            m = m | res[mxId]["masks"]
    else:
        m = res[mxId]["masks"]


    mask = np.where(m,0,255).astype(np.uint8)
    maskPaste = np.where(m,255,0).astype(np.uint8)

    mask = Image.fromarray(mask)
    mask = mask.resize(size)
    mask2 = np.array(mask)

    # scan the mask horizontally from the bottom
    # figure out where the sky starts from the horizon
    for j in reversed(range(size[1])):
        if np.max(mask2[j,:]) == 0:
            lowerBound = j
            break

    # save them to the dump folder, 
    # for parallel calculate in other processes
    maskPaste = Image.fromarray(maskPaste)
    maskPaste = maskPaste.resize(size)

    im.paste(maskPaste,[0,0],mask)
    mask.save(maskFilename)
    maskPaste.save(maskPasteFilename)

    return mask, lowerBound


class MeteorScan:
    def __init__(self, nProcess):
        self.nProcess = nProcess
        self.fps = 8
        self.gridSize = 32
        self.blankVideoLen = 2.5
        self.rectColor = "red"

    def _extract(self, mp4Filename):
        print("extract images")
        # print("mkdir dump")
        os.system("mkdir dump")
        # print("mkdir scanned")
        os.system("mkdir scanned")
        # print("mkdir output")
        os.system("mkdir output")
        # print("rm -Rf dump/*")
        os.system("rm -Rf dump/*")
        mp4R8Filename = mp4Filename.replace(".mp4", ".r%d.mp4" % self.fps)
        targetFilename = os.path.basename(mp4Filename.replace(".mp4","_%5d.png"))
        if not os.path.exists(mp4R8Filename):
            print("convert to %d fps" % self.fps)
            # print("ffmpeg -hide_banner -loglevel error -i %s -r %d %s " % (mp4Filename, self.fps, mp4R8Filename))
            os.system("ffmpeg -hide_banner -loglevel error -i %s -r %d %s " % (mp4Filename, self.fps, mp4R8Filename))
       
        print("dump video to images")
        # print("ffmpeg -hide_banner -loglevel error -i %s dump/%s" %(mp4R8Filename, targetFilename))
        os.system("ffmpeg -hide_banner -loglevel error -i %s dump/%s" %(mp4R8Filename, targetFilename))
        #print("rm -f %s " % mp4R8Filename)
        #os.system("rm -f %s " % mp4R8Filename)


        targetFilenamePattern = mp4Filename.replace(".mp4","_*.png")
        imFilenames = sorted(list(glob.glob("dump/%s" % targetFilenamePattern)))

        mask, lowerBound = generate_mask(imFilenames[0])
        
        return imFilenames, lowerBound - 10

    def _mp_scan(self, srcImFilenames, mp4Filename, start, nSize, lowerBound):
        size = None
        scanned = []
        maskFilename = srcImFilenames[0][:-4]+"_mask.png"
        maskPasteFilename = srcImFilenames[0][:-4]+"_maskPaste.png"
        mask = Image.open(maskFilename)
        maskPaste = Image.open(maskPasteFilename)

        # print("processing: "+mp4Filename)
        imFilenames = srcImFilenames[start:start+nSize]
        nImages = len(imFilenames)

        lsd = cv2.createLineSegmentDetector(0)

        # build cyclic cache for images
        npCyclicCache = np.zeros([int(self.fps*3), mask.size[1], mask.size[0]])
        for i in range(nImages):
            imFilename = imFilenames[i]
            if imFilename.find("_mask")>0:
                continue
            npIm = np.array(Image.open(imFilename).convert("L"))
            for j in range(self.fps * 3):
                npCyclicCache[j] = npIm
            break


        for i in range(nImages):
            imFilename = imFilenames[i]
            if imFilename.find("_mask")>0:
                continue
            # print(imFilename)
            found = False
            imColor = Image.open(imFilename)
            draw = ImageDraw.Draw(imColor)
            im = Image.open(imFilename).convert("L")
            im.paste(maskPaste, [0,0], mask)

            # get average before update take place
            npAveragedCyclicCache = np.average(npCyclicCache, axis=0)
            npIm = np.array(im)
            npCyclicCache[i % int(self.fps*3)] = npIm

            #im.save(imFilename[:-4]+"_masked.png")

            # img = cv2.imread(imFilename[:-4]+"_masked.png" ,0)
            img = cv2.imread(imFilename ,0)
            lsd = cv2.createLineSegmentDetector(0)
            lines = lsd.detect(img)[0]

            def line_to_rect(aLine):
                xs = list(sorted([int(aLine[0]), int(aLine[2])]))
                ys = list(sorted([int(aLine[1]), int(aLine[3])]))
                return xs[0], ys[0], xs[1], ys[1]

            drawn_img = img
            if type(lines) != type(None):
                drawn_img = lsd.drawSegments(img,lines)
                lines = np.squeeze(lines)
                for j in range(lines.shape[0]):
                    # filter by its location
                    if lines[j][1] < lowerBound:
                        # filter by line segment length
                        dist = np.linalg.norm(lines[j,:2] - lines[j,2:])
                        if dist <= 14:
                            continue
                        
                        # filter by lightness changes
                        x1, y1, x2, y2 = line_to_rect(lines[j])
                        x1 = max(0, x1-1)
                        x2 = min(im.size[0]-1, x2+1)
                        y1 = max(0, y1-1)
                        y2 = min(im.size[1]-1, y2+1)
                        diff = np.abs(npAveragedCyclicCache[y1:y2,x1:x2] - npIm[y1:y2,x1:x2]).max()
                        if diff < 40:
                            continue

                        rawPosition = float(os.path.basename(imFilename)[:-4].split("_")[-1])
                        print("detected meteor at %.2fs , lightness diff: %.2f, %s" % (rawPosition/self.fps, diff, str(line_to_rect(lines[j]))))
                        found = True
                        scanned.append(imFilename)
                        break
            # os.system("rm %s" % (imFilename[:-4]+"_masked.png"))


            if found:
                # imColor.save(imFilename.replace("dump", "scanned"))
                cv2.imwrite(imFilename.replace("dump", "scanned"), drawn_img)

        return scanned

    def _scan(self, mp4Filename):
        srcImFilenames, lowerBound = self._extract(mp4Filename)
        nImages = len(srcImFilenames)
        batchSize = int(math.ceil(float(nImages) / float(self.nProcess)))
        futures = []
        with ProcessPoolExecutor(max_workers=self.nProcess) as executor:
            for i in range(self.nProcess):
                startPos = i * batchSize
                endPos = min((i+1) * batchSize, nImages)
                nSize = endPos - startPos
                print("split: ", nImages, i, startPos, nSize)
                future = executor.submit(self._mp_scan, srcImFilenames, mp4Filename, startPos, nSize, lowerBound)
                futures.append(future)

            for i in range(self.nProcess):
                futures[i].result()

    def _cut(self, mp4Filename):
        targetFilenamePattern = mp4Filename.replace(".mp4","_*.png")
        imFilenames = sorted(list(glob.glob("scanned/%s" % targetFilenamePattern)))

        rawPositions = []
        for i in range(len(imFilenames)):
            imFilename = imFilenames[i]
            rawPositions.append([1., float(os.path.basename(imFilename)[:-4].split("_")[-1])])

        dbscan = DBSCAN(eps=self.blankVideoLen*float(self.fps), min_samples=1)
        clusterLabels = dbscan.fit_predict(rawPositions)
        cuts = {}
        for i in range(len(clusterLabels)):
            if str(clusterLabels[i]) in cuts:
                cuts[str(clusterLabels[i])].append(rawPositions[i][1] / float(self.fps))
            else:
                cuts[str(clusterLabels[i])] = [rawPositions[i][1] / float(self.fps)]


        for cutPositions in cuts:
            startPos = int(np.array(cuts[cutPositions]).min()-self.blankVideoLen)
            endPos   = int(np.array(cuts[cutPositions]).max()+self.blankVideoLen)

            outputVideoFilename = "output/"+mp4Filename.replace(".mp4", "_"+str(cutPositions)+".mp4")
            # print("ffmpeg -hide_banner -loglevel error -y -i %s -ss %d -to %d  -pix_fmt yuv420p -c:v h264 -c:a aac %s" %(mp4Filename, startPos, endPos, outputVideoFilename))
            os.system("ffmpeg -hide_banner -loglevel error -y -i %s -ss %d -to %d -pix_fmt yuv420p -c:v h264 -c:a aac  %s" %(mp4Filename, startPos, endPos, outputVideoFilename))

    def scanAndCut(self, mp4Filename):
        self._scan(mp4Filename)
        self._cut(mp4Filename)

if __name__ == "__main__":
    mp4Filename = sys.argv[1]
    nProcess = 1
    if len(sys.argv)>2:
        nProcess = int(sys.argv[2])
    MeteorScan(nProcess = nProcess).scanAndCut(mp4Filename)
