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
```shell
$ python3 meteor_scan.py C.mp4 4
meteor_scan v2.2.0

                    ▄▄▄▄
       ▀██       ▄▄    ██     █    ▄█        ▄▄   ▄█▄   ██
  ██    ▄▄▄      ██▄█ ▄█▀    ▄█▄   █  ▄      ▀▀▀    █   ██
      ▄██▀        ▀██▄▀    ▄██▀▀ ▄█▀▄▄▀██▄        ▄ █ █▄██           ▄▄▀
▄▄    ██▄██      ▄█▀▄██▀  ▀▀██ ▄ ▀ ▀▀▀ ▄ ▀  ██  █▀█ █ █ ██      ▄▄▄█▀
██   ███  ▄▄   ▄███▀▀█▄    █▀█    ▄ █  █▀       ▀██▀█   ██  ▄▄█▀▀
█ ▄  ▄█▀████   ▀▀  ▄███▀    ██   ▀▀  ▄█▀      ▄ ▄███▄▄  ██  ▀▀
██▀  █▀ ▀  ▀▄▄     ▄█▀      ██   ▄▄███████▄ ▄█▀ ▀▀  ▀▀▀ ██
            ▀▀    ██████▀                   ▀           ██

filename: C.mp4
split(s): 4

It is highly recommended to input video with size around 1920x1080
High resolution slows down the entire process.

1. create directories
Following sub-directories will be created:
meteor_scan
    ├── dump
    │   └── ...
    ├── output
    │   └── ...
    └── scanned
        └── ...

2. extract video to images
clean up the previous extraction [y/N] ? y
2.1. convert video to 8 fps with ffmpeg
2.2. dump video to images with ffmpeg

3. find sky background
Processed Image saved successfuly in your current working directory.

4. detect following 4 split(s) in parallel
         frames  splitId startPos splitFrames
split:     3277        0        0      820
split:     3277        1      820      820
split:     3277        2     1640      820
split:     3277        3     2460      817
detected at:  314.00s, position:     [1496, 477, 1511, 486]
detected at:   14.88s, position:     [1849, 523, 1865, 530]
detected at:   31.75s, position:     [1482, 295, 1498, 302]
detected at:   31.88s, position:     [1504, 308, 1525, 317]
detected at:   40.75s, position:     [1163, 459, 1175, 468]
detected at:   40.88s, position:     [1177, 470, 1191, 481]
detected at:   41.00s, position:     [1193, 482, 1207, 493]
detected at:   41.12s, position:     [1207, 497, 1221, 508]
detected at:  279.00s, position:       [982, 373, 993, 383]
detected at:  279.12s, position:      [996, 386, 1008, 397]
detected at:  177.62s, position:     [1483, 287, 1499, 294]
detected at:  390.62s, position:     [1036, 596, 1046, 607]
detected at:  294.25s, position:       [923, 446, 933, 457]
detected at:  294.38s, position:       [935, 454, 947, 468]
detected at:  294.50s, position:       [947, 473, 957, 483]
detected at:  404.50s, position:     [1738, 340, 1756, 348]

5. cut videos
ffmpeg -hide_banner -loglevel error -y -i C.mp4 -ss 12 -to 17  -pix_fmt yuv420p -c:v h264 -c:a aac output/C_0.mp4
ffmpeg -hide_banner -loglevel error -y -i C.mp4 -ss 29 -to 34  -pix_fmt yuv420p -c:v h264 -c:a aac output/C_1.mp4
ffmpeg -hide_banner -loglevel error -y -i C.mp4 -ss 38 -to 43  -pix_fmt yuv420p -c:v h264 -c:a aac output/C_2.mp4
ffmpeg -hide_banner -loglevel error -y -i C.mp4 -ss 100 -to 105  -pix_fmt yuv420p -c:v h264 -c:a aac output/C_3.mp4
ffmpeg -hide_banner -loglevel error -y -i C.mp4 -ss 175 -to 180  -pix_fmt yuv420p -c:v h264 -c:a aac output/C_4.mp4
ffmpeg -hide_banner -loglevel error -y -i C.mp4 -ss 276 -to 281  -pix_fmt yuv420p -c:v h264 -c:a aac output/C_5.mp4
ffmpeg -hide_banner -loglevel error -y -i C.mp4 -ss 291 -to 297  -pix_fmt yuv420p -c:v h264 -c:a aac output/C_6.mp4
ffmpeg -hide_banner -loglevel error -y -i C.mp4 -ss 311 -to 316  -pix_fmt yuv420p -c:v h264 -c:a aac output/C_7.mp4
ffmpeg -hide_banner -loglevel error -y -i C.mp4 -ss 388 -to 393  -pix_fmt yuv420p -c:v h264 -c:a aac output/C_8.mp4
ffmpeg -hide_banner -loglevel error -y -i C.mp4 -ss 402 -to 407  -pix_fmt yuv420p -c:v h264 -c:a aac output/C_9.mp4

6. detection completed!
cut videos are saved to "output" directory
```

Enjoy!

![captured](https://raw.githubusercontent.com/Erickrus/meteor_scan/main/scanned/meteor01_00028.png) 
