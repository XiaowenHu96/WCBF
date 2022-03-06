## What does WCBF do

The program takes input a video stream and outputs a video stream.
The even frames are the original frame, and the odd frames are the
processed frame with background detected and set to black. 
Hence WCBF outputs twice as many frames as it reads.
The output data can then be piped into tools like ffmpeg for further process
and feed into a virtual webcam

## Example: Background Blur

### Initialize Submodule
`git submodule update --init --recursive`

### Initialize Virtual Camera
`sudo modprobe v4l2loopback devices=1`
Form now on the example assume webcam is at `/dev/video0` and v4l2 device is
created at `/dev/video1`. (Check with `v4l2-ctl --list-devices`)

### Blur
The idea is to
1. Feed webcam into WCBF
2. Feed output of WCBF into another ffmpeg
3. Blur the even frames, transparent the odd frames, finally overlay and output to `/dev/video1`

So we have three commands chained together. The actual command is shown below.
Noted that ffmpeg counts frame starting at one.
```
ffmpeg -f v4l2 -i /dev/video0 -pix_fmt rgb24 -f rawvideo -r 30 - 2>/dev/null\
    | python3 main.py -s 640x480 -f rgb24 \
    |  ffmpeg -thread_queue_size 32 -f rawvideo -framerate 60 -pix_fmt rgb24 -video_size 640x480 -i -\
    -filter_complex "[0:v]select='mod(n-1\,2)',boxblur=10[a];[0:v]select='not(mod(n-1\,2))',colorkey=0x000000:0.01:0[b];[a][b]overlay,format=yuv420p,fps=30" -f v4l2 /dev/video2
```
