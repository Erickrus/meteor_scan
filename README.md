# meteor_scan
Scan through video for meteor 

# Installation
Install python packages based on requirements.txt: 
`pip3 install -r requirements.txt`

Also install ffmpeg with: 
`apt install -y ffmpeg`

Finally, download deeplabv3_xception65_ade20k.h5 from following page:
https://pixellib.readthedocs.io/en/latest/video_ade20k.html

# Searching for meteors
to run this program
`
python3 meteor_scan.py meteor01.mp4
`

If you want to speed up, add parallelism in the end (e.g. 4)
`
python3 meteor_scan.py meteor01.mp4 4
`

Enjoy!

![captured](https://raw.githubusercontent.com/Erickrus/meteor_scan/main/scanned/meteor01_00028.png) 
