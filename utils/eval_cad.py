import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
import random
import warnings
from glob import glob
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from plyfile import PlyData
from pathlib import Path
from multiprocessing import Pool
from chamfer_distance import ChamferDistance

N_POINTS = 2000
NUM_TRHEADS = 36


def find_files(folder, extension):
    return sorted([Path(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(extension)])


def read_ply(path):
    with open(path, 'rb') as f:
        plydata = PlyData.read(f)
        x = np.array(plydata['vertex']['x'])
        y = np.array(plydata['vertex']['y'])
        z = np.array(plydata['vertex']['z'])
        vertex = np.stack([x, y, z], axis=1)
    return vertex


def _pairwise_CD(sample_pcs, ref_pcs, batch_size):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    chamfer_dist = ChamferDistance()
    pbar = tqdm(range(N_sample), desc="Computing Chamfer Distances")

    for i in pbar:
        sample = sample_pcs[i]  # (N, 3)
        cd_list = []
        for j in range(0, N_ref, batch_size):
            ref_batch = ref_pcs[j:min(j + batch_size, N_ref)]  # (B, N, 3)
            bsz = ref_batch.size(0)
            sample_expand = sample.unsqueeze(0).expand(bsz, -1, -1).contiguous()  # (B, N, 3)

            dl, dr, _, _ = chamfer_dist(sample_expand, ref_batch)
            cd = dl.mean(dim=1) + dr.mean(dim=1)  # (B,)
            cd_list.append(cd)

        cd_list = torch.cat(cd_list, dim=0)  # (N_ref,)
        all_cd.append(cd_list.unsqueeze(0))  # (1, N_ref)

    all_cd = torch.cat(all_cd, dim=0)  # (N_sample, N_ref)
    return all_cd


def compute_avg_cd(sample_pcs, ref_pcs, batch_size):
    all_dist = _pairwise_CD(sample_pcs, ref_pcs, batch_size)  # (N_sample, N_ref)
    min_cd_per_sample, _ = torch.min(all_dist, dim=1)  # (N_sample,)

    avg_cd = min_cd_per_sample.mean()
    median_cd = min_cd_per_sample.median()

    return {
        'Avg-CD': avg_cd.item(),
        'Median-CD': median_cd.item()
    }


def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, in_unit_sphere, resolution=28):
    sample_grid_var = entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(pclouds, grid_resolution, in_sphere=False):
    epsilon = 10e-4
    bound = 1 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1) * 2
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5 * 2
                grid[i, j, k, 1] = j * spacing - 0.5 * 2
                grid[i, j, k, 2] = k * spacing - 0.5 * 2

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[np.linalg.norm(grid, axis=1) <= 0.5]

    return grid, spacing


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    return e_sum - ((e1 + e2) / 2.0)


def downsample_pc(points, n):
    sample_idx = random.sample(list(range(points.shape[0])), n)
    return points[sample_idx]


def normalize_pc(points):
    scale = np.max(np.abs(points))
    points = points / scale
    return points


def collect_pc(cad_folder):
    pc_path = find_files(os.path.join(cad_folder, 'pcd'), 'final_pcd.ply')
    if len(pc_path) == 0:
        return []
    pc_path = pc_path[-1]
    pc = read_ply(pc_path)
    if pc.shape[0] > N_POINTS:
        pc = downsample_pc(pc, N_POINTS)
    pc = normalize_pc(pc)
    return pc


def compute_pairwise_cd(sample_pcs, ref_pcs, batch_size):
    """
    Compute one-to-one Chamfer Distance between sample and reference point clouds.
    Assumes sample_pcs and ref_pcs have the same number of point clouds and correspond to each other.
    
    Args:
        sample_pcs: Generated/reconstructed point clouds (N, num_points, 3)
        ref_pcs: Ground truth point clouds (N, num_points, 3)
        batch_size: Batch size for computation
    
    Returns:
        Dictionary with CD metrics
    """
    assert sample_pcs.shape[0] == ref_pcs.shape[0], "Sample and reference must have same number of point clouds"
    
    N = sample_pcs.shape[0]
    chamfer_dist = ChamferDistance()
    all_cd = []
    
    pbar = tqdm(range(0, N, batch_size), desc="Computing Pairwise Chamfer Distances")
    
    for i in pbar:
        end_idx = min(i + batch_size, N)
        sample_batch = sample_pcs[i:end_idx]  # (B, num_points, 3)
        ref_batch = ref_pcs[i:end_idx]        # (B, num_points, 3)
        
        dl, dr, _, _ = chamfer_dist(sample_batch, ref_batch)
        cd = dl.mean(dim=1) + dr.mean(dim=1)  # (B,)
        all_cd.append(cd)
    
    all_cd = torch.cat(all_cd, dim=0)  # (N,)
    
    avg_cd = all_cd.mean()
    median_cd = all_cd.median()
    std_cd = all_cd.std()
    
    return {
        'Pairwise-Avg-CD': avg_cd.item(),
        'Pairwise-Median-CD': median_cd.item(),
        'Pairwise-Std-CD': std_cd.item(),
        'All-CD': all_cd.cpu().numpy()  # Return all individual CD values for further analysis
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", type=str)
    parser.add_argument("--real", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--n_test", type=int, default=1988)
    parser.add_argument("--multi", type=int, default=3)
    parser.add_argument("--times", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    print("n_test: {}, multiplier: {}, repeat times: {}".format(args.n_test, args.multi, args.times))
    if args.output is None:
        args.output = args.fake + '_cad_results.txt'

    # Load reference point clouds
    ref_pcs = []
    project_folders = sorted(glob(args.real + '/*/'))
    load_iter = Pool(NUM_TRHEADS).imap(collect_pc, project_folders)
    for pc in tqdm(load_iter, total=len(project_folders), desc="Loading real point clouds"):
        if len(pc) > 0:
            ref_pcs.append(pc)
    ref_pcs = np.stack(ref_pcs, axis=0)
    print("Loaded real point clouds:", ref_pcs.shape)

    # Load generated point clouds
    sample_pcs = []
    project_folders = sorted(glob(args.fake + '/*/'))
    load_iter = Pool(NUM_TRHEADS).imap(collect_pc, project_folders)
    for pc in tqdm(load_iter, total=len(project_folders), desc="Loading fake point clouds"):
        if len(pc) > 0:
            sample_pcs.append(pc)
    sample_pcs = np.stack(sample_pcs, axis=0)
    print("Loaded fake point clouds:", sample_pcs.shape)

    # Evaluation
    fp = open(args.output, "w")
    result_list = []
    for i in range(args.times):
        print(f"Iteration {i}...")
        select_idx = random.sample(list(range(len(sample_pcs))), int(args.multi * args.n_test))
        rand_sample_pcs = sample_pcs[select_idx]

        select_idx = random.sample(list(range(len(ref_pcs))), args.n_test)
        rand_ref_pcs = ref_pcs[select_idx]

        jsd = jsd_between_point_cloud_sets(rand_sample_pcs, rand_ref_pcs, in_unit_sphere=False)

        with torch.no_grad():
            rand_sample_pcs = torch.tensor(rand_sample_pcs).cuda()
            rand_ref_pcs = torch.tensor(rand_ref_pcs).cuda()
            result = compute_avg_cd(rand_sample_pcs, rand_ref_pcs, batch_size=args.batch_size)

        result.update({"JSD": jsd})

        print(result)
        print(result, file=fp)
        result_list.append(result)

    # Average results
    avg_result = {}
    for k in result_list[0].keys():
        avg_result["avg-" + k] = np.mean([x[k] for x in result_list])
    print("Average result:")
    print(avg_result)
    print(avg_result, file=fp)
    fp.close()


if __name__ == '__main__':
    main()
