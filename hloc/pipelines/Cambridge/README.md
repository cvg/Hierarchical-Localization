# Cambridge Landmarks dataset

## Installation

Download the dataset from the [PoseNet project page](http://mi.eng.cam.ac.uk/projects/relocalisation/):
```bash
export dataset=datasets/cambridge
wget https://www.repository.cam.ac.uk/bitstream/handle/1810/251342/KingsCollege.zip -P $dataset
wget https://www.repository.cam.ac.uk/bitstream/handle/1810/251340/OldHospital.zip -P $dataset
wget https://www.repository.cam.ac.uk/bitstream/handle/1810/251294/StMarysChurch.zip -P $dataset
wget https://www.repository.cam.ac.uk/bitstream/handle/1810/251336/ShopFacade.zip -P $dataset
wget https://www.repository.cam.ac.uk/bitstream/handle/1810/251291/GreatCourt.zip -P $dataset
unzip $dataset/KingsCollege.zip -d $dataset
unzip $dataset/OldHospital.zip -d $dataset
unzip $dataset/StMarysChurch.zip -d $dataset
unzip $dataset/ShopFacade.zip -d $dataset
unzip $dataset/GreatCourt.zip -d $dataset
```

Download the SIFT SfM models, courtesy of Torsten Sattler:
```bash
export fileid=1esqzZ1zEQlzZVic-H32V6kkZvc4NeS15
export filename=$dataset/CambridgeLandmarks_Colmap_Retriangulated_1024px.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$fileid" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$fileid" -O $filename && rm -rf /tmp/cookies.txt
unzip $filename -d $dataset
```

## Pipeline

```bash
python3 -m hloc.pipelines.Cambridge.pipeline
```
