from pathlib import Path
import argparse
from tqdm import tqdm


def generate(dataset, sequence, outputs):
    h, w = 1024, 1024
    images_dir = dataset / 'images/'
    query_list_name = '{}_queries_with_intrinsics.txt'
    intrinsics_name = 'intrinsics/{}_intrinsics.txt'
    intrinsics = {}
    for side in ['left', 'right', 'rear']:
        with open(dataset / intrinsics_name.format(side), 'r') as f:
            fx = f.readline().split()[1]
            fy = f.readline().split()[1]
            cx = f.readline().split()[1]
            cy = f.readline().split()[1]
            assert fx == fy
            params = ['SIMPLE_RADIAL', w, h, fx, cx, cy, 0.0]
            intrinsics[side] = [str(p) for p in params]

    query_file = open(outputs / query_list_name.format(sequence), 'w')
    paths = sorted([p for p in Path(images_dir, sequence).glob('**/*.jpg')])
    for p in tqdm(paths):
        name = str(Path(p).relative_to(images_dir))
        side = Path(p).parent.name
        query_file.write(' '.join([name]+intrinsics[side])+'\n')
    query_file.close()


def main(dataset, outputs):
    sequences = [p.name for p in (dataset/'images').iterdir() if p.is_dir()]
    for seq in sequences:
        if seq == 'overcast-reference':
            continue
        generate(dataset, seq, outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, required=True)
    args = parser.parse_args()
    main(**args.__dict__)
