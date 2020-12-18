import sys
from pathlib import Path
import subprocess
import logging
import torch

from zipfile import ZipFile
import os

from ..utils.base_model import BaseModel

dir_path = Path(__file__).parent / '../../third_party/deep-image-retrieval'

sys.path.append(str(dir_path))

from dirtorch import nets as nets
from dirtorch.utils import common

# This doesn't work directly, need to comment lines 17, 19, and 20 from `dirtorch/extract_features.py`
#from dirtorch.extract_features import load_model

def load_model(path, iscuda=True):
    checkpoint = common.load_checkpoint(path, iscuda)
    net = nets.create_model(pretrained="", **checkpoint['model_options'])
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    if 'pca' in checkpoint:
        net.pca = checkpoint.get('pca')
    return net

class DIR(BaseModel):
    default_conf = {
        'model_name': 'Resnet-101-AP-GeM',
        'checkpoint': dir_path / 'dirtorch/data',
        'whiten': 'Landmarks_clean',
        'whitenp': 0.25,
        'whitenv': None,
        'whitenm': 1.0,
        'pooling': 'gem',
        'gemp': 3,
    }
    required_inputs = ['image']

    def _init(self, conf):
        dir_models = {
            'Resnet-101-AP-GeM': 'https://docs.google.com/uc?export=download&id=1UWJGDuHtzaQdFhSMojoYVQjmCXhIwVvy',
        }

        model_file = dir_path / 'dirtorch/data' / str(conf['model_name']+'.pt')

        if not model_file.exists():
            model_file.parent.mkdir(exist_ok=True)
            cmd = ['wget', '--no-check-certificate', '-r', dir_models[conf['model_name']],
                   '-O', str(model_file)+'.zip']
            ret = subprocess.call(cmd)
            zf = ZipFile(str(model_file)+'.zip', 'r')
            zf.extractall(model_file.parent)
            zf.close()
            os.remove(str(model_file)+'.zip')
            if ret != 0:
                logging.warning(
                    f'Cannot download the DIR model with `{cmd}`.')
                exit(ret)

        self.net = load_model(conf['checkpoint'] / str(conf['model_name']+'.pt'))

        if conf['whiten']:
            self.net.pca = self.net.pca[conf['whiten']]
        else:
            self.net.pca = None

    def _forward(self, data):
        desc = self.net(data)
        return {
            'global_descriptor': desc[None],
        }
