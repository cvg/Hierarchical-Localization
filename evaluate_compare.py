from pathlib import Path

from hloc.pipelines.Cambridge.utils import evaluate


if __name__ == '__main__':

    suffix = '2015'

    outputs = Path('outputs/TourEiffel') / suffix
    results1 = outputs / 'results.txt'
    results2 = outputs / 'results_covis_clustering.txt'
    gt_dir = Path('datasets/toureiffel') / suffix

    evaluate(
        gt_dir / 'empty_all', results1,
        gt_dir / 'list_query.txt', ext='.txt'
    )

    evaluate(
        gt_dir / 'empty_all', results2,
        gt_dir / 'list_query.txt', ext='.txt'
    )
