import gdown

url = 'https://drive.google.com/uc?id=1Vtwk71rFhFQLY39e-guPPuG5OLevonhR'

gdown.download(url, 'checkpoints/', quiet=False)
