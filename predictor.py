import os
import cv2
import sys
import gdown
import hashlib
import torch
import torchvision
import numpy as np
import onnxruntime as ort
from pathlib import Path
from skimage import transform 
from contextlib import redirect_stdout
from MODNet.src.models.modnet import MODNet

def get_ort_session_initializer(path : str):
    def initializer():
        return ort.InferenceSession(str(path), providers=ort.get_available_providers())
    return initializer

def u2net_ort_norm(frame : np.array) -> np.array:
    new_frame = np.zeros((frame.shape[0], frame.shape[1], 3))
    frame = frame / np.max(frame)

    if frame.shape[2] == 1:
        new_frame[:, :, 0] = (frame[:, :, 0] - 0.485) / 0.229
        new_frame[:, :, 1] = (frame[:, :, 0] - 0.485) / 0.229
        new_frame[:, :, 2] = (frame[:, :, 0] - 0.485) / 0.229
    else:
        new_frame[:, :, 0] = (frame[:, :, 0] - 0.485) / 0.229
        new_frame[:, :, 1] = (frame[:, :, 1] - 0.456) / 0.224
        new_frame[:, :, 2] = (frame[:, :, 2] - 0.406) / 0.225

    new_frame = new_frame.transpose((2, 0, 1))

    return new_frame
    
def u2net_resize(frame : np.array) -> np.array:
    output_size = 320
    frame = transform.resize(frame, (output_size, output_size), mode="constant")
    return frame

""" Input frame must be preprocessed """
def u2net_ort_inferencer(ort_session: ort.InferenceSession, frame : np.array) -> np.array:
    input_frame = np.expand_dims(frame, 0).astype(np.float32)

    ort_inputs = {ort_session.get_inputs()[0].name: input_frame}
    ort_outs = ort_session.run(None, ort_inputs)

    d1 = ort_outs[0]
    pred = d1[:, 0, :, :]
    max_pred, min_pred = (np.max(pred), np.min(pred))
    pred = (pred - min_pred) / (max_pred - max_pred)
    pred = np.squeeze(pred)
    return pred

def get_modnet_torch_initializer(path : str):
    def initializer():
        modnet = MODNet(backbone_pretrained=False)
        modnet = torch.nn.DataParallel(modnet)
        GPU = True if torch.cuda.device_count() > 0 else False
        if GPU:
            print('Use GPU...')
            modnet = modnet.cuda()
            modnet.load_state_dict(torch.load(path))
        else:
            print('Use CPU...')
            modnet.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        modnet.eval()
        return modnet
    return initializer

def get_modnet_torch_norm():
    tf = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    def norm(frame : np.array):
        return tf(frame)[None, :, :, :]
    return norm
    
def modnet_resize(frame : np.array) -> np.array:
    ref_size = 512
    h, w = frame.shape[0:2]
    if w >= h:
        rh = ref_size
        rw = int(w / h * ref_size)
    else:
        rw = ref_size
        rh = int(h / w * ref_size)
    rh = rh - rh % 32
    rw = rw - rw % 32
    # TODO: skimage resize did not work when feed into tensor (type error)
    frame = cv2.resize(frame, (rh, rw), cv2.INTER_AREA)
    return frame

""" Input frame must be preprocessed """
def modnet_torch_inferencer(modnet, tensor) -> np.array:
    GPU = True if torch.cuda.device_count() > 0 else False
    if GPU:
        tensor = tensor.cuda()
    with torch.no_grad():
        _, _, result = modnet(tensor, True)
    return result[0][0].data.cpu().numpy()

class Predictor:
    def __init__(self, initializer, preprocess_pipeline, inferencer):
        self.model = initializer()
        self.preprocess_pipeline = preprocess_pipeline
        self.inferencer = inferencer

    def predict(self, frame : np.array):
        h, w = frame.shape[0:2]
        for preprocess in self.preprocess_pipeline:
            frame = preprocess(frame)
        pred =  self.inferencer(self.model, frame)
        pred = transform.resize(pred, (h, w), mode="constant")
        return pred


def PredictorBuilder(model_name : str):
    # Download model if not already
    # TODO cleanup: should be a dict
    if model_name == "u2netp":
        md5 = "8e83ca70e441ab06c318d82300c84806"
        url = "https://drive.google.com/uc?id=1tNuFmLv0TSNDjYIkjEdeH1IWKQdUA4HR"
        extension = "onnx"
    elif model_name == "u2net":
        md5 = "60024c5c889badc19c04ad937298a77b"
        url = "https://drive.google.com/uc?id=1tCU5MM1LhRgGou5OpmpjBQbSrYIUoYab"
        extension = "onnx"
    elif model_name == "u2net_human_seg":
        md5 = "c09ddc2e0104f800e3e1bb4652583d1f"
        url = "https://drive.google.com/uc?id=1ZfqwVxu-1XWC1xU1GHIP-FM_Knd_AX5j"
        extension = "onnx"
    elif model_name == "modnet_webcam_portrait_matting":
        md5 = "b60360be61de983191433769658f89cc"
        url = "https://drive.google.com/uc?id=1Nf1ZxeJZJL8Qx9KadcYYyEmmlKhTADxX"
        extension = "ckpt"
    elif model_name == "modnet_photographic_portrait_matting":
        md5 = "63396b0e3fdf0a7ef11dfc850cde2773"
        url = "https://drive.google.com/uc?id=1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz"
        extension = "ckpt"
    else:
        raise Exception("Unable to find model {}."
                "This should probably be a bug. Please report.".format(model_name))

    home = os.path.join("~", ".WebCamVirtualBackground")
    path = Path(home).expanduser() / f"{model_name}.{extension}"
    path.parents[0].mkdir(parents=True, exist_ok=True)

    if not (path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == md5):
        with redirect_stdout(sys.stderr):
            gdown.download(url, str(path), use_cookies=False)

    if model_name == "u2netp":
        initializer = get_ort_session_initializer(path)
        preprocess_pipeline = [u2net_resize, u2net_ort_norm]
        inferencer = u2net_ort_inferencer
    elif model_name == "u2net":
        initializer = get_ort_session_initializer(path)
        preprocess_pipeline = [u2net_resize, u2net_ort_norm]
        inferencer = u2net_ort_inferencer
    elif model_name == "u2net_human_seg":
        initializer = get_ort_session_initializer(path)
        preprocess_pipeline = [u2net_resize, u2net_ort_norm]
        inferencer = u2net_ort_inferencer
    elif model_name == "modnet_webcam_portrait_matting":
        initializer = get_modnet_torch_initializer(path)
        preprocess_pipeline = [modnet_resize, get_modnet_torch_norm()]
        inferencer = modnet_torch_inferencer
    elif model_name == "modnet_photographic_portrait_matting":
        initializer = get_modnet_torch_initializer(path)
        preprocess_pipeline = [modnet_resize, get_modnet_torch_norm()]
        inferencer = modnet_torch_inferencer

    return Predictor(initializer, preprocess_pipeline, inferencer)
