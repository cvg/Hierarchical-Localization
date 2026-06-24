"""Parametrized Aachen Day-Night v1.1 runner for LateMambaGlue OR MambaGlue.

Backward compatible with the earlier nc2/nc4 launches. Pick the matcher with
--matcher_name; pass --n_cross_layers only for latemambaglue.

LateMambaGlue (nc4):
    CUDA_VISIBLE_DEVICES=1 python run_aachen_lmg.py \
        --outputs outputs/aachen_v1.1_nc4 \
        --checkpoint .../latemambaglueMDnc4/checkpoint_best.tar \
        --matcher_name latemambaglue --n_cross_layers 4 --tag nc4

MambaGlue baseline:
    CUDA_VISIBLE_DEVICES=2 python run_aachen_lmg.py \
        --outputs outputs/aachen_v1.1_mg \
        --checkpoint .../sp+mg_megadepth/checkpoint_best.tar \
        --matcher_name mambaglue --tag mg
"""

import argparse
from pathlib import Path

from hloc import (
    extract_features,
    localize_sfm,
    match_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation,
)


def run(args):
    dataset = args.dataset
    images = dataset / "images_upright/"
    sift_sfm = dataset / "3D-models/aachen_v_1_1"

    outputs = args.outputs
    outputs.mkdir(exist_ok=True, parents=True)
    name = args.matcher_name
    sfm_pairs = outputs / f"pairs-db-covis{args.num_covis}.txt"
    loc_pairs = outputs / f"pairs-query-netvlad{args.num_loc}.txt"
    reference_sfm = outputs / f"sfm_superpoint-open+{name}-{args.tag}"
    results = (
        outputs
        / f"Aachen-v1.1_hloc_superpoint-open+{name}-{args.tag}_netvlad{args.num_loc}.txt"
    )

    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_mg_aachen"]

    model = {
        "name": name,
        "features": "superpoint",
        "checkpoint": args.checkpoint,
        "filter_threshold": args.filter_threshold,
    }
    if args.n_layers is not None:
        model["n_layers"] = args.n_layers
    if args.n_cross_layers is not None:
        model["n_cross_layers"] = args.n_cross_layers
    matcher_conf = {"output": f"matches-superpoint-{name}-{args.tag}", "model": model}

    print(f"[{args.tag}] matcher={name} ckpt={args.checkpoint} "
          f"n_cross_layers={args.n_cross_layers}")

    features = extract_features.main(feature_conf, images, outputs, as_half=True)

    pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=args.num_covis)
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )
    triangulation.main(
        reference_sfm, sift_sfm, images, sfm_pairs, features, sfm_matches
    )

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(
        global_descriptors, loc_pairs, args.num_loc,
        query_prefix="query", db_model=reference_sfm,
    )
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], outputs
    )
    localize_sfm.main(
        reference_sfm,
        dataset / "queries/*_time_queries_with_intrinsics.txt",
        loc_pairs,
        features,
        loc_matches,
        results,
        covisibility_clustering=False,
    )
    print(f"\n[{args.tag}] DONE -> {results}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", type=Path, default="datasets/aachen_v1.1")
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--matcher_name", type=str, default="latemambaglue",
                   choices=["latemambaglue", "mambaglue"])
    p.add_argument("--n_cross_layers", type=int, default=None,
                   help="latemambaglue only; omit for mambaglue")
    p.add_argument("--n_layers", type=int, default=None,
                   help="omit to use the matcher's own default")
    p.add_argument("--filter_threshold", type=float, default=0.1)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--num_covis", type=int, default=20)
    p.add_argument("--num_loc", type=int, default=50)
    run(p.parse_args())