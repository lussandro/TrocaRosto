#!/usr/bin/python
"""
Troca de rosto em tempo real com rede neural (InsightFace inswapper_128).

    python ao_vivo.py <rosto.jpg> [indice_camera] [--virtual]

Diferente do webcam.py (que so cola olhos/nariz/boca), aqui a rede gera o
rosto inteiro da foto acompanhando a sua expressao. Precisa de placa NVIDIA
para ficar fluido: na CPU roda a ~0,5 quadro por segundo.
Com --virtual o resultado vai para a camera virtual (v4l2loopback no Linux,
OBS Virtual Camera no Windows). Tecla q ou ESC fecha a janela.
"""

import hashlib
import os
import sys
import time
import urllib.request

import cv2
import onnxruntime
import insightface
from insightface.app import FaceAnalysis

MODEL_URL = "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx"
MODEL_SHA256 = "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "inswapper_128.onnx")
DET_SIZE = (320, 320)


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("Baixando modelo (529 MB) de %s ..." % MODEL_URL)
    tmp = MODEL_PATH + ".part"
    urllib.request.urlretrieve(MODEL_URL, tmp)
    h = hashlib.sha256()
    with open(tmp, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != MODEL_SHA256:
        os.remove(tmp)
        sys.exit("Modelo baixado com checksum errado: %s" % h.hexdigest())
    os.replace(tmp, MODEL_PATH)


def biggest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def main():
    virtual = "--virtual" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--virtual"]
    if len(args) < 1:
        sys.exit(__doc__)

    ensure_model()
    # Carrega as DLLs de CUDA/cuDNN instaladas pelo pip (onnxruntime-gpu[cuda,cudnn]).
    onnxruntime.preload_dlls()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in onnxruntime.get_available_providers()]

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                       providers=providers)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    swapper = insightface.model_zoo.get_model(MODEL_PATH, providers=providers)
    print("Executando em: %s" % swapper.session.get_providers()[0])

    src_img = cv2.imread(args[0])
    if src_img is None:
        sys.exit("Nao foi possivel abrir a imagem %s" % args[0])
    src_faces = app.get(src_img)
    if not src_faces:
        sys.exit("Nenhum rosto encontrado em %s" % args[0])
    src = biggest(src_faces)

    cam_index = int(args[1]) if len(args) > 1 else 0
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        sys.exit("Nao foi possivel abrir a camera %d" % cam_index)

    vcam = None
    if virtual:
        import pyvirtualcam
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vcam = pyvirtualcam.Camera(width=width, height=height, fps=20)
        print("Camera virtual: %s" % vcam.device)

    fps, t0 = 0.0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            sys.exit("Falha ao ler quadro da camera %d" % cam_index)

        faces = app.get(frame)
        out = frame if not faces else swapper.get(frame, biggest(faces), src,
                                                  paste_back=True)

        if vcam is not None:
            # Sem espelhar: o Meet espelha so a sua previa, os outros veem normal.
            vcam.send(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        preview = cv2.flip(out, 1)

        t1 = time.time()
        fps = 0.9 * fps + 0.1 / max(t1 - t0, 1e-6)
        t0 = t1
        cv2.putText(preview, "%.1f fps" % fps, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("TrocaRosto", preview)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    if vcam is not None:
        vcam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
