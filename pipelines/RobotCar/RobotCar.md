# RobotCar dataset

This is a pre-release of scripts that run hloc on the RobotCar Seasons dataset. Multi-camera localization will follow later.

## Commands
```bash
ROBOTCAR=/path/to/dataset

python robotcar_to_colmap.py \
    --nvm $ROBOTCAR/3D-models/all-merged/all.nvm \
    --database $ROBOTCAR/3D-models/overcast-reference.db \
    --output outputs/sfm_sift
    
python -m hloc.pairs_from_covisibility \
    --model outputs/sfm_sift \
    --output pairs/pairs-db-covis20.txt \
    --num_matched 20

python -m hloc.extract_features --image_dir $ROBOTCAR/images/ --export_dir outputs/

python -m hloc.match_features --pairs pairs/pairs-db-covis20.txt --export_dir outputs/

python -m hloc.triangulation \
    --sfm_dir outputs/sfm_superpoint+superglue \
    --reference_sfm_model outputs/sfm_sift \
    --image_dir $ROBOTCAR/images/ \
    --pairs pairs/pairs-db-covis20.txt \
    --features outputs/feats-superpoint-n4096-r1024.h5 \
    --matches outputs/feats-superpoint-n4096-r1024_matches-superglue_pairs-db-covis20.h5

python -m hloc.match_features \
    --pairs pairs/pairs-query-netvlad20-percam-perloc.txt \
    --export_dir outputs/
    
python robotcar_generate_query_list.py \
    --dataset $ROBOTCAR \
    --outputs outputs/
    
python localize.py \
    --reference_sfm outputs/sfm_superpoint+superglue/model \
    --queries "outputs/*_queries_with_intrinsics.txt" \
    --retrieval pairs/pairs-query-netvlad20-percam-perloc.txt \
    --features outputs/feats-superpoint-n4096-r1024.h5 \
    --matches outputs/feats-superpoint-n4096-r1024_matches-superglue_pairs-query-netvlad20-percam-perloc.h5 \
    --results outputs/RobotCar_hloc_superpoint+superglue_netvlad20-percam-perloc.txt \
    [--v2]
```
