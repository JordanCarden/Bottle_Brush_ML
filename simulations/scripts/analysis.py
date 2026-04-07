import os
import subprocess
import re
import ast
import csv
import time
import MDAnalysis as mda
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull


def cluster_points(points, cutoff):
    points = np.array(points)
    d2 = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    connectivity = d2 <= cutoff**2
    n_components, labels = connected_components(
        connectivity, directed=False, connection="strong"
    )
    return [np.where(labels == i)[0].tolist() for i in range(n_components)]


def is_compact(points, rg_threshold=10.0):
    points = np.array(points)
    if len(points) < 2:
        return True
    com = np.mean(points, axis=0)
    rg = np.sqrt(np.mean(np.sum((points - com) ** 2, axis=1)))
    return rg < rg_threshold


def find_cluster_frame(u, cutoff=40.0, rg_threshold=60.0, min_frames=100):
    """Return the first frame with a single compact cluster.

    If the trajectory has fewer than ``min_frames`` frames, ``None`` is
    returned so the polymer can be analyzed later once the simulation
    finishes. When a cluster never forms, ``np.nan`` is returned.
    """

    n_frames = len(u.trajectory)

    if n_frames < min_frames:
        return None

    def has_single_compact_cluster(frame):
        u.trajectory[frame]
        atoms = u.select_atoms("resname LIG")
        if not atoms.residues:
            return False
        centers = [res.atoms.center_of_mass() for res in atoms.residues]
        clusters = cluster_points(centers, cutoff)
        return len(clusters) == 1 and is_compact(centers, rg_threshold)

    if not has_single_compact_cluster(n_frames - min_frames):
        return np.nan
    candidate = np.nan
    low, high = 0, n_frames - 100
    while low <= high:
        mid = (low + high) // 2
        if has_single_compact_cluster(mid):
            candidate = mid
            high = mid - 1
        else:
            low = mid + 1
    return candidate


def extract_input_list(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    match = re.search(r"input_list\s*=\s*(\[[\s\S]*?\])", content, re.DOTALL)
    if not match:
        return None
    list_str = match.group(1)
    try:
        return ast.literal_eval(list_str)
    except Exception:
        return None


def compute_convex_hull_areas(xyz_file):
    with open(xyz_file, "r") as file:
        lines = file.readlines()
    rows_per_frame = int(lines[0])
    total_frames = len(lines) // (rows_per_frame + 2)
    areas = []
    for frame in range(1, total_frames):
        start = frame * (rows_per_frame + 2)
        frame_data = np.array(
            [
                list(map(float, line.split()[1:4]))
                for line in lines[start + 2:start + 2 + rows_per_frame]
            ]
        )
        hull = ConvexHull(frame_data[:, :2])
        areas.append(hull.volume)
    avg_area = np.mean(areas) if areas else 0
    std_area = np.std(areas, ddof=1) if len(areas) > 1 else 0
    area_file = os.path.join(os.path.dirname(xyz_file), "area.txt")
    with open(area_file, "w") as out:
        for area in areas:
            out.write(f"{area}\n")
        out.write(f"{avg_area}\n")
        out.write(f"{std_area}")
    return area_file


def get_frames_values(data_path, cluster_frame):
    start_frame = 830
    end_frame = 899
    frames_file = os.path.join(data_path, "frames.txt")
    with open(frames_file, "w") as f:
        f.write(f"{start_frame} {end_frame}")
    with open(frames_file, "r") as f:
        frames_line = f.readline().strip()
    return frames_line.split()


def run_vmd(data_path, analysis_tcl_path):
    subprocess.run(
        ["vmd", "-dispdev", "text", "-e", analysis_tcl_path],
        cwd=data_path,
        check=True,
    )


def get_area_values(data_path):
    area_file = os.path.join(data_path, "area.txt")
    if os.path.isfile(area_file):
        with open(area_file, "r") as f:
            return [line.strip() for line in f if line.strip() != ""]
    return ["area.txt not found"]


def get_rg_values(data_path):
    rg_file = os.path.join(data_path, "rg.txt")
    rg_avg = []
    rg_std = []
    if os.path.isfile(rg_file):
        with open(rg_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    rg_avg.append(parts[0])
                    rg_std.append(parts[1])
        return rg_avg, rg_std
    return (["rg.txt not found"], [])


def get_rdf_values(data_path):
    rdf_file = os.path.join(data_path, "rdf.txt")
    rdf_vals = []
    coord_vals = []
    if os.path.isfile(rdf_file):
        with open(rdf_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    rdf_vals.append(parts[1])
                    coord_vals.append(parts[2])
        return rdf_vals, coord_vals
    return (["rdf.txt not found"], [])


def find_rdf_peak_and_coord_min(rdf_vals, coord_vals):
    rdf_peak = "NA"
    coord_at_min = "NA"
    try:
        rdf_floats = [float(x) for x in rdf_vals]
        if rdf_floats:
            index_max = max(
                range(len(rdf_floats)), key=lambda i: rdf_floats[i]
            )
            rdf_peak = rdf_floats[index_max]
            local_min_index = None
            for i in range(index_max + 1, len(rdf_floats) - 3):
                if all(
                    rdf_floats[i] < rdf_floats[i - offset]
                    and rdf_floats[i] < rdf_floats[i + offset]
                    for offset in range(1, 4)
                ):
                    local_min_index = i
                    break
            if local_min_index is not None:
                coord_at_min = float(coord_vals[local_min_index])
            else:
                coord_at_min = "NA"
    except Exception:
        rdf_peak = "NA"
        coord_at_min = "NA"
    return rdf_peak, coord_at_min


def process_polymer(polymer_dir, analysis_tcl_path, total_columns):
    polymer_name = os.path.basename(polymer_dir)
    data_path = os.path.join(polymer_dir, "20sim")
    modbond_file = os.path.join(polymer_dir, "modifybond.py")
    input_list = extract_input_list(modbond_file)
    psf_file = os.path.join(data_path, "20_interfaceafterpgn.psf")
    dcd_file = os.path.join(data_path, "system.dcd")

    try:
        u = mda.Universe(psf_file, dcd_file)
    except Exception:
        row = [polymer_name, str(input_list), "NaN", "NaN"]
        row.extend(["NaN"] * (total_columns - 4))
        return row

    cluster_frame = find_cluster_frame(u)

    if cluster_frame is None:
        return None

    if np.isnan(cluster_frame):
        row = [polymer_name, str(input_list), "NaN", "NaN"]
        row.extend(["NaN"] * (total_columns - 4))
        return row

    frames = get_frames_values(data_path, cluster_frame)
    run_vmd(data_path, analysis_tcl_path)
    xyz_file = os.path.join(data_path, "xyz.xyz")
    if os.path.isfile(xyz_file):
        compute_convex_hull_areas(xyz_file)
    area_values = get_area_values(data_path)
    rg_avg, rg_std = get_rg_values(data_path)
    rdf_vals, coord_vals = get_rdf_values(data_path)
    rdf_peak, coord_at_min = find_rdf_peak_and_coord_min(rdf_vals, coord_vals)

    row = [polymer_name, str(input_list)]
    row.extend(frames)
    row.extend(area_values)
    row.extend(rg_avg)
    row.extend(rg_std)
    row.extend(rdf_vals)
    row.extend(coord_vals)
    row.append(rdf_peak)
    row.append(coord_at_min)

    if len(row) < total_columns:
        row.extend(["NaN"] * (total_columns - len(row)))
    return row


def assemble_analysis_data(polymers_dir, analysis_tcl_path, skip_polymers):
    header = []
    header.append("Name")
    header.append("Input List")
    header.append("Start Frame")
    header.append("End Frame")
    for i in range(1, 71):
        header.append(f"Area Frame {i}")
    header.append("Area AVG")
    header.append("Area STD")
    for i in range(1, 71):
        header.append(f"RG Frame {i}")
    header.append("RG AVG")
    for i in range(1, 71):
        header.append(f"RG STD Frame {i}")
    header.append("RG STD")
    for i in range(200):
        r_val = format(i * 0.1, ".1f")
        header.append(f"RDF {r_val}")
    for i in range(200):
        r_val = format(i * 0.1, ".1f")
        header.append(f"Coordination {r_val}")
    header.append("RDF Peak")
    header.append("Coordination at Minimum")

    total_columns = len(header)
    rows = []
    for polymer in sorted(os.listdir(polymers_dir)):
        if polymer in skip_polymers:
            continue
        polymer_dir = os.path.join(polymers_dir, polymer)
        if os.path.isdir(polymer_dir):
            row = process_polymer(
                polymer_dir, analysis_tcl_path, total_columns
            )
            if row is not None:
                rows.append(row)
    return header, rows


def main():
    base_dir = os.path.dirname(__file__)
    polymers_dir = os.path.abspath(os.path.join(base_dir, "..", "polymers"))
    analysis_tcl_path = os.path.join(base_dir, "analysis.tcl")
    analysis_csv = os.path.abspath(
        os.path.join(base_dir, "..", "data", "analysis.csv")
    )
    existing_polymers = set()
    if os.path.exists(analysis_csv):
        with open(analysis_csv, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            try:
                next(reader)
            except StopIteration:
                pass
            for row in reader:
                if row:
                    existing_polymers.add(row[0])
    header, rows = assemble_analysis_data(
        polymers_dir, analysis_tcl_path, existing_polymers
    )
    if os.path.exists(analysis_csv):
        with open(analysis_csv, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for row in rows:
                writer.writerow(row)
    else:
        with open(analysis_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total time: {time.time() - start:.2f} seconds")
