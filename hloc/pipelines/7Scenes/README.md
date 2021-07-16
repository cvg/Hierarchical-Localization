# 7-Scenes dataset

## Installation

Download the images from the [7-Scenes project page](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/):
```bash
export dataset=datasets/7scenes
for scene in chess fire heads office pumpkin redkitchen stairs; \
do wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/$scene.zip -P $dataset \
&& unzip $dataset/$scene.zip -d $dataset && unzip $dataset/$scene/'*.zip' -d $dataset/$scene; done
```

Download the SIFT SfM models and DenseVLAD image pairs, courtesy of Torsten Sattler:
```bash
function download {
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -rf /tmp/cookies.txt
unzip $2 -d $dataset && rm $2;
}
download 1cu6KUR7WHO7G4EO49Qi3HEKU6n_yYDjb $dataset/7scenes_sfm_triangulated.zip
download 1IbS2vLmxr1N0f3CEnd_wsYlgclwTyvB1 $dataset/7scenes_densevlad_retrieval_top_10.zip
```

## Pipeline

```bash
python3 -m hloc.pipelines.7Scenes.pipeline [--use_dense_depth]
```
