# import sys
# replace the path 
# sys.path.append(r'path/to/vietocr')
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import yolov5


YOLOV5_MODEL_FILE_PATH = r'model\bestnew.pt'

# load and return yolov5 model
def load_yolov5():
    return yolov5.load(YOLOV5_MODEL_FILE_PATH)



# load and return viet
def load_vietocr():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'
    predictor = Predictor(config)
    return predictor
