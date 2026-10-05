#!/usr/bin/python
"""
Troca de rosto em tempo real pela webcam.

    python webcam.py <rosto.jpg> [indice_camera] [--virtual]

O rosto da imagem e aplicado sobre o seu rosto em cada quadro da webcam.
Com --virtual o resultado tambem vai para a camera virtual (v4l2loopback),
para usar no Meet/Zoom. Tecla q ou ESC fecha a janela.
"""

import sys
import time

import cv2
import dlib
import numpy

from faceswap import (detector, predictor, read_im_and_landmarks,
                      transformation_from_points, get_face_mask, warp_im,
                      correct_colours, ALIGN_POINTS)

# Fator de reducao do quadro para a deteccao (o detector e a parte lenta).
DETECT_SCALE = 0.5


def frame_landmarks(frame):
    small = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
    rects = detector(small, 0)
    if len(rects) == 0:
        return None
    # ponytail: com varios rostos, usa o maior (mais perto da camera)
    r = max(rects, key=lambda r: r.area())
    s = 1.0 / DETECT_SCALE
    rect = dlib.rectangle(int(r.left() * s), int(r.top() * s),
                          int(r.right() * s), int(r.bottom() * s))
    return numpy.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])


def swap(frame, landmarks1, im2, landmarks2, mask2):
    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])
    warped_mask = warp_im(mask2, M, frame.shape)
    combined_mask = numpy.max([get_face_mask(frame, landmarks1), warped_mask],
                              axis=0)
    warped_im2 = warp_im(im2, M, frame.shape)
    warped_corrected_im2 = correct_colours(frame, warped_im2, landmarks1)
    out = frame * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    return numpy.clip(out, 0, 255).astype(numpy.uint8)


def main():
    virtual = "--virtual" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--virtual"]
    if len(args) < 1:
        sys.exit(__doc__)
    im2, landmarks2 = read_im_and_landmarks(args[0])
    mask2 = get_face_mask(im2, landmarks2)

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

        landmarks1 = frame_landmarks(frame)
        out = frame if landmarks1 is None else swap(frame, landmarks1,
                                                    im2, landmarks2, mask2)

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
