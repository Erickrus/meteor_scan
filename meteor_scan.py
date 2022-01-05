# -*- coding: utf-8 -*-

'''
Meteor Scan 

Author: Hu, Ying-Hao (hyinghao@hotmail.com)
Version: 1.0.0
Last modification date: 2022-01-05
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
    Leverage "outlier detection" approach to find out the meteors and their trails.
    
    Assume each input video is a continuous stream for the dark sky
    As the sky background will rotate/shift slightly through the time, 
    build a rolling average for each 32x32 piece within 2-3 seconds as the normal sample
    
    Then, check whether anything will happen through time,
    which is abnormal from it's previous averaged normal sample
    
    Finally, use spatial cluster to merge all continuous frames
    cut them from the original video

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
import glob
from PIL import Image, ImageDraw
import math
import numpy as np
import sys

from sklearn.cluster import DBSCAN
from concurrent.futures import ProcessPoolExecutor

class MeteorScan:
    def __init__(self, nProcess):
        self.nProcess = nProcess
        self.fps = 8
        self.gridSize = 32
        self.blankVideoLen = 2.5
        self.rectColor = "red"

    def _extract(self, mp4Filename):
        print("mkdir dump")
        os.system("mkdir dump")
        print("mkdir scanned")
        os.system("mkdir scanned")
        print("mkdir output")
        os.system("mkdir output")
        print("rm -Rf dump/*")
        os.system("rm -Rf dump/*")
        mp4R8Filename = mp4Filename.replace(".mp4", ".r%d.mp4" % self.fps)
        targetFilename = os.path.basename(mp4Filename.replace(".mp4","_%5d.png"))
        print("ffmpeg -i %s -r %d %s " % (mp4Filename, self.fps, mp4R8Filename))
        os.system("ffmpeg -i %s -r %d %s " % (mp4Filename, self.fps, mp4R8Filename))
        print("ffmpeg -i %s dump/%s" %(mp4R8Filename, targetFilename))
        os.system("ffmpeg -i %s dump/%s" %(mp4R8Filename, targetFilename))
        print("rm -f %s " % mp4R8Filename)
        os.system("rm -f %s " % mp4R8Filename)


        targetFilenamePattern = mp4Filename.replace(".mp4","_*.png")
        imFilenames = sorted(list(glob.glob("dump/%s" % targetFilenamePattern)))
        
        return imFilenames

    def _mp_scan(self, srcImFilenames, mp4Filename, start, nSize):
        size = None
        scanned = []
        print("processing: "+mp4Filename)
        imFilenames = srcImFilenames[start:start+nSize]
        nImages = len(imFilenames)
        # build backgrounds
        if size == None:
            size = Image.open(imFilenames[0]).size
        grids = {}
        for i in range(3*int(self.fps)):
            imFilename = imFilenames[i]
            im = Image.open(imFilename).convert("L")
            im = np.array(im)
            for x in range(size[0]//self.gridSize):
                for y in range(size[1] //self.gridSize):
                    gridId = "%d_%d" % (x, y)
                    piece = im[y*self.gridSize:(y+1)*self.gridSize,x*self.gridSize:(x+1)*self.gridSize]
                    if gridId in grids:
                        grids[gridId].append(piece.tolist())
                    else:
                        grids[gridId] = [piece.tolist()]

        
        for i in range(nImages):
            imFilename = imFilenames[i]
            print(imFilename)
            found = False
            imColor = Image.open(imFilename)
            draw = ImageDraw.Draw(imColor)
            im = Image.open(imFilename).convert("L")
            im = np.array(im)
            for x in range(size[0]//self.gridSize):
                for y in range(size[1] //self.gridSize):
                    gridId = "%d_%d" % (x, y)
                    piece = im[y*self.gridSize:(y+1)*self.gridSize,x*self.gridSize:(x+1)*self.gridSize]
                    v = np.array(grids[gridId])
                    v = np.mean(v, axis=0)
                    diff = (piece - v).max()
                    totDiff = np.abs(piece - v)
                    totDiff = totDiff[totDiff > diff - 20].size

                    if diff > 52 and totDiff > 6:
                        scanned.append(imFilename)
                        print("%s,%d,%d" % (imFilename,x*self.gridSize,y*self.gridSize))
                        draw.rectangle((x*self.gridSize,y*self.gridSize,(x+1)*self.gridSize,(y+1)*self.gridSize), outline =self.rectColor)
                        found = True
                    del grids[gridId][0]
                    grids[gridId].append(piece.tolist())
            if found:
                imColor.save(imFilename.replace("dump", "scanned"))

        return scanned

    def _scan(self, mp4Filename):
        srcImFilenames = self._extract(mp4Filename)
        nImages = len(srcImFilenames)
        batchSize = int(math.ceil(float(nImages) / float(self.nProcess)))
        futures = []
        with ProcessPoolExecutor(max_workers=self.nProcess) as executor:
            for i in range(self.nProcess):
                startPos = i * batchSize
                endPos = min((i+1) * batchSize, nImages)
                nSize = endPos - startPos
                print(nImages, i, startPos, nSize)
                future = executor.submit(self._mp_scan, srcImFilenames, mp4Filename, startPos, nSize)
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
        for i in clusterLabels:
            if str(clusterLabels[i]) in cuts:
                cuts[str(clusterLabels[i])].append(rawPositions[i][1] / float(self.fps))
            else:
                cuts[str(clusterLabels[i])] = [rawPositions[i][1] / float(self.fps)]

        for cutPositions in cuts:
            startPos = int(np.array(cuts[cutPositions]).min()-self.blankVideoLen)
            endPos   = int(np.array(cuts[cutPositions]).max()+self.blankVideoLen)

            outputVideoFilename = "output/"+mp4Filename.replace(".mp4", "_"+str(cutPositions)+".mp4")
            print("ffmpeg -y -i %s -ss %d -to %d -c copy %s" %(mp4Filename, startPos, endPos, outputVideoFilename))
            os.system("ffmpeg -y -i %s -ss %d -to %d -c copy %s" %(mp4Filename, startPos, endPos, outputVideoFilename))

    def scanAndCut(self, mp4Filename):
        self._scan(mp4Filename)
        self._cut(mp4Filename)

if __name__ == "__main__":
    mp4Filename = sys.argv[1]
    nProcess = 1
    if len(sys.argv)>2:
        nProcess = int(sys.argv[2])
    MeteorScan(nProcess = nProcess).scanAndCut(mp4Filename)
    


