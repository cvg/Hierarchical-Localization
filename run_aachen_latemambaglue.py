"""End-to-end Aachen Day-Night v1.1 visual localization
   with open-SuperPoint + LateMambaGlue (trained in glue-factory).

This mirrors hloc's stock hloc/pipelines/Aachen_v1_1/pipeline.py, with three
swaps from the SuperGlue default:
  - extractor -> superpoint_open_aachen      (matches glue-factory training)
  - matcher   -> superpoint+latemambaglue    (your trained checkpoint)
  - output names tagged so runs don't collide with a baseline run

Place this at the hloc repo root (or anywhere `import hloc` resolves) and run:

    python run_aachen_latemambaglue.py \
        --dataset datasets/aachen_v1.1 \
        --outputs outputs/aachen_v1.1_latemambaglue \
        --num_covis 20 --num_loc 50

Prerequisites (see the chat for full setup):
  1. hloc/matchers/latemambaglue.py        (wrapper, provided)
  2. hloc/extractors/superpoint_open.py    (wrapper, provided)
  3. confs entries registered in hloc/match_features.py and
     hloc/extract_features.py (provided)
  4. glue-factory importable (`pip install -e .` in the glue-factory repo)
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
    sift_sfm = dataset / "3D-models/aachen_v_1_1"  # COLMAP model shipped with v1.1

    outputs = args.outputs
    outputs.mkdir(exist_ok=True, parents=True)
    sfm_pairs = outputs / f"pairs-db-covis{args.num_covis}.txt"
    loc_pairs = outputs / f"pairs-query-netvlad{args.num_loc}.txt"

    # ---- the swaps vs. the stock SuperGlue pipeline -------------------------
    reference_sfm = outputs / "sfm_superpoint-open+latemambaglue"
    results = (
        outputs
        / f"Aachen-v1.1_hloc_superpoint-open+latemambaglue_netvlad{args.num_loc}.txt"
    )

    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_open_aachen"]   # swap 1
    matcher_conf = match_features.confs["superpoint+latemambaglue"]   # swap 2
    # -------------------------------------------------------------------------

    # 1) Local features for every db + query image (open SuperPoint)
    features = extract_features.main(feature_conf, images, outputs, as_half=True)

    # 2) Reference SfM: covisible db pairs -> LateMambaGlue match -> triangulate
    pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=args.num_covis)
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )
    triangulation.main(
        reference_sfm, sift_sfm, images, sfm_pairs, features, sfm_matches
    )

    # 3) Global descriptors (NetVLAD) -> top-k retrieval pairs, query vs db
    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        args.num_loc,
        query_prefix="query",
        db_model=reference_sfm,
    )

    # 4) Match query<->db with LateMambaGlue, then PnP + RANSAC localization
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
        covisibility_clustering=False,  # standard for Aachen
    )

    print(f"\nDone.\nEstimated query poses: {results}")
    print("Upload that file to https://www.visuallocalization.net/ for day/night accuracy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, default="datasets/aachen_v1.1")
    parser.add_argument(
        "--outputs", type=Path, default="outputs/aachen_v1.1_latemambaglue"
    )
    parser.add_argument("--num_covis", type=int, default=20,
                        help="db image pairs for SfM triangulation")
    parser.add_argument("--num_loc", type=int, default=50,
                        help="retrieved db pairs per query (50 = best Aachen results)")
    run(parser.parse_args())