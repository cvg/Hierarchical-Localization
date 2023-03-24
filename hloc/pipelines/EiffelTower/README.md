# Eiffel Tower underwater dataset

## Installation

Download the dataset from the [SEANOE](https://www.seanoe.org/data/00810/92226/):
```bash
export dataset=datasets/eiffeltower
export IDs=( "98240" "98289" "98314" "98356" "98357" )
for id in "${IDs[@]}"; do
wget https://www.seanoe.org/data/00810/92226/data/$id.zip -P $dataset \
&& unzip $dataset/$id.zip -d $dataset && rm $dataset/$id.zip; done
```


## Pipeline

```bash
python3 -m hloc.pipelines.EiffelTower.pipeline
```
