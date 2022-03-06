import sys
import cv2
import argparse
import numpy as np
from concurrent import futures

import predictor

models_shortname_map = {
        "u2netp" : "u2netp",
        "u2net"  : "u2net",
        "u2net_human" : "u2net_human_seg",
        "modnet_webcam" : "modnet_webcam_portrait_matting",
        "modnet_photo" : "modnet_photographic_portrait_matting"
}

def filter(img: np.array, mask: np.array)  -> np.array:
    result = np.zeros(img.shape)
    mask = (mask[:,:,np.newaxis] > 0.5).astype(int)
    result[:,:,0] = img[:, :, 0] * mask[:, :, 0]
    result[:,:,1] = img[:, :, 1] * mask[:, :, 0]
    result[:,:,2] = img[:, :, 2] * mask[:, :, 0]
    return result.astype(np.uint8)

def error_exit(str : str):
    print(str, file=sys.stderr)
    exit(-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TODO Some discription here")
    parser.add_argument('-s', '--video_size',required=True,\
            help="Input resolution: e.g. 640x480") 
    parser.add_argument('-f', '--color_fmt', required=True,\
            choices=["rgb24"],\
            help="Input color format: e.g. rgb24")
    parser.add_argument('-m', '--model', default='modnet_webcam',\
            choices=list(models_shortname_map.keys()))
    parser.add_argument('--cv2_realtime_test', action='store_true', default=False,\
            help="Do not output data, inseadt, display a cv2 window.")

    args = parser.parse_args()

    # check valid size
    f_width, f_height = (0,0)
    try:
        f_width, f_height = list(map(int, args.video_size.split('x')))
        if not (f_width > 0 and f_height > 0):
            raise Exception()
    except Exception:
        error_exit("Invalid video size {}. See --help.".format(args.video_size))

    # get channels
    f_dim =  3 if args.color_fmt == "rgb24" else 1

    # Initialize Predictor
    thread_pool = futures.ThreadPoolExecutor(max_workers=1)
    predictor = predictor.PredictorBuilder(models_shortname_map[args.model])

    f_shape = (f_height, f_width, f_dim)
    mask = None
    task = None
    while True:
        raw_data = sys.stdin.buffer.read(f_height * f_width * f_dim)
        frame = np.frombuffer(raw_data, np.uint8).reshape(f_shape)
        if (task == None):
            task = thread_pool.submit(predictor.predict, frame)
        if (mask is None):
            mask = task.result(2000)
            task = thread_pool.submit(predictor.predict, frame)
        if (task != None and task.done()):
            mask = task.result()
            task = thread_pool.submit(predictor.predict, frame)

        frame = filter(frame, mask)
        if not(args.cv2_realtime_test):
            sys.stdout.buffer.write(raw_data)
            sys.stdout.buffer.write(frame.reshape(f_height * f_width * f_dim))
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('CV2 Realtime Test', frame)
            c = cv2.waitKey(1)
            if c == 27:
                break
