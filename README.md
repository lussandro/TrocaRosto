# Troca rosto - Face swap

Objetivo:

- Realizar a identificação e troca de rosto em imagens através de duas imagens, uma origem, outra de destino.
- Como rodar a aplicação? ﻿Execute o comando faceswap.py origem.jpg destino.jpg e isso vai gerar uma imagem chamada resultado.jpg

- O modelo "shape_predictor_68_face_landmarks.dat" é lido da pasta `shape_predictor_68_face_landmarks/` do próprio projeto, não precisa mover.

- Abra o arquivo "/core/recognizer.py " e altere as linha de 63 a 67 com os dados obtidos no site: faceplusplus.com conforme a instrução.

```python
def landmarks_by_face__(image):
    url = 'https://api-us.faceplusplus.com/facepp/v3/detect'
    params = {
        'api_key': 'sua chave aqui',
        'api_secret': 'seu segredo aqui',
```        

# Tempo real pela webcam

Dois scripts, os dois com janela de prévia (tecla `q` ou `ESC` fecha) e opção `--virtual` para mandar o vídeo para uma câmera virtual (Meet, Zoom, etc.).

- `webcam.py` — mesmo método do `faceswap.py` (dlib): cola olhos, sobrancelhas, nariz e boca da foto. Leve (~10 fps na CPU), mas não fica parecido com a pessoa da foto.
- `ao_vivo.py` — rede neural (InsightFace `inswapper_128`): gera o rosto inteiro acompanhando sua expressão. Fica parecido de verdade, mas precisa de placa NVIDIA (na CPU roda a ~0,5 fps). Na primeira execução baixa os modelos (~530 MB em `models/` e ~280 MB em `~/.insightface/`).

## Windows com placa NVIDIA (`ao_vivo.py`)

Pré-requisitos: driver NVIDIA atualizado (CUDA 12), Python 3.10 a 3.12, Git e [OBS Studio](https://obsproject.com/) (fornece a câmera virtual "OBS Virtual Camera"). Placas da série Kepler (GT 710/730, GTX 650–780) não são suportadas pelo CUDA 12.

```bat
git clone https://github.com/lussandro/TrocaRosto.git
cd TrocaRosto
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install insightface opencv-python pyvirtualcam
pip uninstall -y onnxruntime
pip install "onnxruntime-gpu[cuda,cudnn]"
python ao_vivo.py "Exemplos\Barack Obama.jpg" --virtual
```

O `insightface` instala o `onnxruntime` de CPU, que conflita com o `onnxruntime-gpu` — por isso o `uninstall` antes. O script mostra `Executando em: CUDAExecutionProvider` quando está usando a placa; se mostrar `CPUExecutionProvider`, a GPU não foi carregada (veja as mensagens de erro do onnxruntime acima dessa linha). No Meet, escolha a câmera **OBS Virtual Camera**.

## Linux

```bash
python3 -m venv .venv
.venv/bin/pip install opencv-python numpy dlib pyvirtualcam insightface onnxruntime
sudo apt install v4l2loopback-dkms
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="TrocaRosto" exclusive_caps=1
.venv/bin/python webcam.py "Exemplos/Barack Obama.jpg" --virtual
```

No Meet, escolha a câmera **TrocaRosto**.

# Exemplos e resultados:


- Barack Obama
![](https://github.com/chaos4455/TrocaRosto/blob/master/Exemplos/Barack%20Obama.jpg?raw=true)

- Ronaldinho Gaúcho
![](https://github.com/chaos4455/TrocaRosto/blob/master/Exemplos/Ronaldinho%20Ra%C3%BAcho.jpg?raw=true)

- Ronaldo Obama ou Obama Gaúcho
![](https://github.com/chaos4455/TrocaRosto/blob/master/Exemplos/Ronaldo%20Obama.jpg?raw=true)

# Acha que pode acrescentar algo interessante, fazer alguma melhoria ou correção? Fique a vontade!

- Updates:

30/01/2019 05:22
Feito upload inicial do projeto. 

