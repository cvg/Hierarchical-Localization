# Eiffel Tower underwater dataset

## Installation

Download the dataset from the [SEANOE](https://www.seanoe.org/data/00810/92226/):
```bash
export dataset=datasets/eiffeltower
export IDs=( "98240" "98289" "98314" "98356" "98357" )
for id in "${IDs[@]}"; do
wget https://www.seanoe.org/data/00810/92226/data/$id.zip -P $dataset \
&& unzip $dataset/$id.zip -d $dataset && rm $dataset/$id.zip; done
mkdir $dataset/global/images
for year in 2015 2016 2018 2020; do
for image in `ls $dataset/$year/images/`; do
ln -s ../../$year/images/$image $dataset/global/images/;
done; done
```


## Pipeline

```bash
python3 -m hloc.pipelines.EiffelTower.pipeline
```
