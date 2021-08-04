import sys
from pathlib import Path
import subprocess
import torch
from zipfile import ZipFile
import os
import sklearn

from ..utils.base_model import BaseModel

dir_path = Path(__file__).parent / '../../third_party/deep-image-retrieval'
sys.path.append(str(dir_path))
os.environ['DB_ROOT'] = ''  # required by dirtorch

from dirtorch.utils import common  # noqa: E402
from dirtorch.extract_features import load_model  # noqa: E402

# The DIR model checkpoints (pickle files) include sklearn.decomposition.pca,
# which has been deprecated in sklearn v0.24
# and must be explicitly imported with `from sklearn.decomposition import PCA`.
# This is a hacky workaround to maintain forward compatibility.
sys.modules['sklearn.decomposition.pca'] = sklearn.decomposition._pca


class DIR(BaseModel):
    default_conf = {
        'model_name': 'Resnet-101-AP-GeM',
        'checkpoint_dir': dir_path / 'dirtorch/data',
        'whiten_name': 'Landmarks_clean',
        'whiten_params': {
            'whitenp': 0.25,
            'whitenv': None,
            'whitenm': 1.0,
        },
        'pooling': 'gem',
        'gemp': 3,
    }
    required_inputs = ['image']

    dir_models = {
        'Resnet-101-AP-GeM': 'https://docs.google.com/uc?export=download&id=1UWJGDuHtzaQdFhSMojoYVQjmCXhIwVvy',
    }

    def _init(self, conf):
        checkpoint = conf['checkpoint_dir'] / str(conf['model_name']+'.pt')
        if not checkpoint.exists():
            checkpoint.parent.mkdir(exist_ok=True)
            link = self.dir_models[conf['model_name']]
            cmd = ['wget', '--no-check-certificate', '-r', link,
                   '-O', str(checkpoint)+'.zip']
            subprocess.run(cmd, check=True)
            zf = ZipFile(str(checkpoint)+'.zip', 'r')
            zf.extractall(checkpoint.parent)
            zf.close()
            os.remove(str(checkpoint)+'.zip')

        self.net = load_model(checkpoint, False)  # first load on CPU
        if conf['whiten_name']:
            assert conf['whiten_name'] in self.net.pca

    def _forward(self, data):
        image = data['image']
        assert image.shape[1] == 3
        mean = self.net.preprocess['mean']
        std = self.net.preprocess['std']
        image = image - image.new_tensor(mean)[:, None, None]
        image = image / image.new_tensor(std)[:, None, None]

        desc = self.net(image)
        desc = desc.unsqueeze(0)  # batch dimension
        if self.conf['whiten_name']:
            pca = self.net.pca[self.conf['whiten_name']]
            desc = common.whiten_features(
                    desc.cpu().numpy(), pca, **self.conf['whiten_params'])
            desc = torch.from_numpy(desc)

        return {
            'global_descriptor': desc,
        }
