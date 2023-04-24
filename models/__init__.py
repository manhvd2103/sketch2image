import importlib
from models.pix2pix import Pix2PixModel

def create_model(opt):
    model = Pix2PixModel(opt)
    instance = model(opt)
    print('Model [%] was created' % type(instance).__name__)
    return instance