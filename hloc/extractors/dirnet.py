import sys
from pathlib import Path
import subprocess
import logging
import torch

from zipfile import ZipFile
import os

from ..utils.base_model import BaseModel

dirnet_path = Path(__file__).parent / '../../third_party/deep-image-retrieval'

sys.path.append(str(dirnet_path))

from dirtorch import nets as nets
from dirtorch.utils import common


def load_model(path, iscuda):
    checkpoint = common.load_checkpoint(path, iscuda)
    net = nets.create_model(pretrained="", **checkpoint['model_options'])
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    if 'pca' in checkpoint:
        net.pca = checkpoint.get('pca')
    return net


class DIRnet(BaseModel):
    default_conf = {
        'model_name': 'Resnet-101-AP-GeM.pt',
        'checkpoint': dirnet_path / 'dirtorch/data/Resnet-101-AP-GeM.pt',
        'whiten': 'Landmarks_clean',
        'whitenp': 0.25,
        'whitenv': None,
        'whitenm': 1.0,
        'pooling': 'gem',
        'gemp': 3,
        'use_gpu': True,
    }
    required_inputs = ['image']

    def _init(self, conf):
        model_file = dirnet_path / 'dirtorch/data' / conf['model_name']

        if not model_file.exists():
            model_file.parent.mkdir(exist_ok=True)
            cmd = ['wget', '--no-check-certificate', '-r', 'https://docs.google.com/uc?export=download&id=1UWJGDuHtzaQdFhSMojoYVQjmCXhIwVvy',
                   '-O', str(model_file)+'.zip']
            ret = subprocess.call(cmd)
            zf = ZipFile(str(model_file)+'.zip', 'r')
            zf.extractall(model_file.parent)
            zf.close()
            os.remove(str(model_file)+'.zip')
            if ret != 0:
                logging.warning(
                    f'Cannot download the DIRnet model with `{cmd}`.')
                exit(ret)

        self.net = load_model(conf['checkpoint'], iscuda=conf['use_gpu'])

        if conf['whiten']:
            self.net.pca = self.net.pca[conf['whiten']]
        else:
            self.net.pca = None

    def _forward(self, data):
        imgs = data
        desc = self.net(imgs)
        return {
            'global_descriptor': desc[None],
        }
