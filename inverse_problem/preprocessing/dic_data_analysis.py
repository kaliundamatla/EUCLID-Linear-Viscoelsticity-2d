"""
DIC Data Analysis Tool for EUCLID Experiments.

This script analyzes raw experimental data and produces:
1. A comprehensive data summary table
2. Force vs Time visualization
3. DIC point cloud visualization

Usage:
    python dic_data_analysis.py [experiment_number]

    Example: python dic_data_analysis.py 1

If no argument is provided, it will prompt for the experiment number.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Configuration for each experiment based on raw data analysis."""
    name: str
    dic_folder: str
    force_file: str
    force_skip_rows: int
    force_column: int  # 0-indexed
    force_column_alt: Optional[int]  # Alternative force column
    time_column: int  # 0-indexed
    dic_separator: str
    dic_skip_rows: int
    description: str


# Experiment configurations based on raw data analysis
# NOTE: All experiments use 10kN sensor for better accuracy at low loads (~200 N)
EXPERIMENT_CONFIGS = {
    1: ExperimentConfig(
        name="Exp1_Solid_PA6_200N",
        dic_folder="solid_PA6_200N_DIC",
        force_file="solid_PA6_200N_20mins.csv",
        force_skip_rows=1,
        force_column=0,  # 10kN sensor (Col 0: "10kN [kN]")
        force_column_alt=4,  # 50kN sensor (Col 4: "Kraft 50kN [kN]")
        time_column=8,
        dic_separator=",",
        dic_skip_rows=5,
        description="PA6 Solid Rectangular Specimen (200N, 20 mins creep)"
    ),
    2: ExperimentConfig(
        name="Exp2_PA6_LaserCut_Rect",
        dic_folder="RMBC",
        force_file="20241213_EUCLID_lve_pa6_rec_LaserCut_Sand_1Hz.csv",
        force_skip_rows=1,
        force_column=2,  # 10kN sensor (Col 2: "Kraft 10kN [kN]")
        force_column_alt=1,  # 50kN sensor (Col 1: "Kraft 50kN [kN]")
        time_column=8,
        dic_separator=";",
        dic_skip_rows=5,
        description="PA6 Rectangular Laser-Cut Specimen (1Hz cyclic)"
    ),
    3: ExperimentConfig(
        name="Exp3_PA6_WJ_Ellipsoid",
        dic_folder="RMBC",
        force_file="PA6_WJ_ellipsoid.csv",
        force_skip_rows=1,
        force_column=2,  # 10kN sensor (Col 2: "Kraft 10kN [kN]")
        force_column_alt=1,  # 50kN sensor (Col 1: "Kraft 50kN [kN]")
        time_column=8,
        dic_separator=";",
        dic_skip_rows=5,
        description="PA6 Elliptical Hole Water-Jet Cut Specimen"
    ),
    4: ExperimentConfig(
        name="Exp4_PA6_WJ_ThreeHoles",
        dic_folder="RMBC",
        force_file="PA6_WJ_three_holes.csv",
        force_skip_rows=1,
        force_column=2,  # 10kN sensor (Col 2: "Kraft 10kN [kN]")
        force_column_alt=1,  # 50kN sensor (Col 1: "Kraft 50kN [kN]")
        time_column=8,
        dic_separator=";",
        dic_skip_rows=5,
        description="PA6 Three Holes Water-Jet Cut Specimen"
    ),
}


def get_base_path() -> Path:
    """Get the base path for experimental data."""
    # Try to find the correct base path
    possible_paths = [
        Path(__file__).parent.parent / "Real_data" / "experiments",
        Path("../Real_data/experiments"),
        Path(r"c:\Users\undamatl\Nextcloud2\EUCLID\euclid-Python_unda\Pipeline\Lin_viscoelastic_identification\inverse_problem\Real_data\experiments"),
    ]

    for path in possible_paths:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError("Could not find experiments base directory")


def analyze_force_file(force_path: Path, config: ExperimentConfig) -> Dict:
    """Analyze the machine force data file."""
    result = {
        "file_exists": force_path.exists(),
        "file_name": force_path.name,
        "columns": [],
        "n_rows": 0,
        "time_range": (0, 0),
        "force_range_primary": (0, 0),
        "force_range_alt": (0, 0),
        "sampling_rate": 0,
        "time_data": None,
        "force_data": None,
        "force_data_alt": None,
    }

    if not force_path.exists():
        return result

    try:
        # Read force file
        df = pd.read_csv(force_path, sep=';', skiprows=config.force_skip_rows,
                        encoding='latin1', on_bad_lines='skip')

        result["columns"] = list(df.columns)
        result["n_rows"] = len(df)

        # Extract time
        time_col = df.iloc[:, config.time_column]
        time_data = pd.to_numeric(time_col, errors='coerce').dropna().values
        result["time_data"] = time_data
        result["time_range"] = (float(time_data.min()), float(time_data.max()))

        # Calculate sampling rate
        if len(time_data) > 1:
            dt = np.diff(time_data)
            result["sampling_rate"] = 1.0 / np.mean(dt)

        # Extract primary force
        force_col = df.iloc[:, config.force_column]
        force_data = pd.to_numeric(force_col, errors='coerce').dropna().values
        result["force_data"] = force_data
        result["force_range_primary"] = (float(force_data.min()), float(force_data.max()))

        # Extract alternative force if available
        if config.force_column_alt is not None:
            try:
                force_col_alt = df.iloc[:, config.force_column_alt]
                force_data_alt = pd.to_numeric(force_col_alt, errors='coerce').dropna().values
                result["force_data_alt"] = force_data_alt
                result["force_range_alt"] = (float(force_data_alt.min()), float(force_data_alt.max()))
            except:
                pass

    except Exception as e:
        result["error"] = str(e)

    return result


def analyze_dic_folder(dic_path: Path, config: ExperimentConfig) -> Dict:
    """Analyze the DIC data folder."""
    result = {
        "folder_exists": dic_path.exists(),
        "folder_name": dic_path.name,
        "n_files": 0,
        "file_pattern": "",
        "time_range": (0, 0),
        "columns": [],
        "n_nodes_first": 0,
        "x_range": (0, 0),
        "y_range": (0, 0),
        "first_file_data": None,
    }

    if not dic_path.exists():
        return result

    try:
        # Find DIC files
        dic_files = sorted(dic_path.glob("Flächenkomponente 1_*.csv"))
        result["n_files"] = len(dic_files)

        if len(dic_files) == 0:
            return result

        # Extract time values from filenames
        times = []
        for f in dic_files:
            try:
                time_str = f.stem.split('_')[1].split(' ')[0]
                times.append(float(time_str))
            except:
                continue

        if times:
            result["time_range"] = (min(times), max(times))

        # Analyze first file
        first_file = dic_files[0]
        result["file_pattern"] = first_file.name

        df = pd.read_csv(first_file, sep=config.dic_separator,
                        skiprows=config.dic_skip_rows, encoding='utf-8')
        df.columns = df.columns.str.strip()

        result["columns"] = list(df.columns)
        result["n_nodes_first"] = len(df)
        result["first_file_data"] = df

        if 'x' in df.columns and 'y' in df.columns:
            result["x_range"] = (float(df['x'].min()), float(df['x'].max()))
            result["y_range"] = (float(df['y'].min()), float(df['y'].max()))

    except Exception as e:
        result["error"] = str(e)

    return result


def print_summary_table(exp_num: int, config: ExperimentConfig,
                        force_info: Dict, dic_info: Dict):
    """Print a formatted summary table."""

    print("\n" + "="*80)
    print(f"  EXPERIMENT {exp_num} DATA SUMMARY")
    print(f"  {config.description}")
    print("="*80)

    # Table format
    rows = [
        ("Parameter", f"Experiment {exp_num}"),
        ("-"*30, "-"*40),
        ("DIC Folder", dic_info.get("folder_name", "N/A")),
        ("Force File", force_info.get("file_name", "N/A")),
        ("Force Skip Rows", f"{config.force_skip_rows} (Header)"),
        ("Force Column (Primary)", f"Col {config.force_column} ({force_info.get('columns', ['?'])[config.force_column] if len(force_info.get('columns', [])) > config.force_column else '?'})"),
        ("Force Column (Alt)", f"Col {config.force_column_alt}" if config.force_column_alt else "N/A"),
        ("Force Profile", f"{force_info['force_range_primary'][0]:.4f} - {force_info['force_range_primary'][1]:.4f} kN" if force_info.get('force_data') is not None else "N/A"),
        ("Time Column", f"Col {config.time_column}"),
        ("Time Range", f"{force_info['time_range'][0]:.1f} - {force_info['time_range'][1]:.1f} s" if force_info.get('time_data') is not None else "N/A"),
        ("Duration", f"{(force_info['time_range'][1] - force_info['time_range'][0])/60:.1f} minutes" if force_info.get('time_data') is not None else "N/A"),
        ("Sampling Rate", f"~{force_info['sampling_rate']:.2f} Hz" if force_info.get('sampling_rate') else "N/A"),
        ("DIC Separator", f"'{config.dic_separator}' ({'semicolon' if config.dic_separator == ';' else 'comma'})"),
        ("DIC Skip Rows", f"{config.dic_skip_rows} (Header lines)"),
        ("DIC Files Count", f"{dic_info.get('n_files', 0)}"),
        ("DIC Nodes (t=0)", f"{dic_info.get('n_nodes_first', 0):,}"),
        ("DIC X Range", f"{dic_info['x_range'][0]:.2f} - {dic_info['x_range'][1]:.2f} mm" if dic_info.get('x_range') else "N/A"),
        ("DIC Y Range", f"{dic_info['y_range'][0]:.2f} - {dic_info['y_range'][1]:.2f} mm" if dic_info.get('y_range') else "N/A"),
        ("Specimen Width", f"{dic_info['x_range'][1] - dic_info['x_range'][0]:.2f} mm" if dic_info.get('x_range') else "N/A"),
        ("Specimen Height", f"{dic_info['y_range'][1] - dic_info['y_range'][0]:.2f} mm" if dic_info.get('y_range') else "N/A"),
    ]

    # Print table
    for param, value in rows:
        print(f"  {param:<25} | {value}")

    print("="*80)

    # Print column details
    print("\n  Force File Columns:")
    for i, col in enumerate(force_info.get('columns', [])):
        marker = " <-- TIME" if i == config.time_column else ""
        marker = " <-- FORCE (PRIMARY)" if i == config.force_column else marker
        marker = " <-- FORCE (ALT)" if i == config.force_column_alt else marker
        print(f"    Col {i}: {col}{marker}")

    print("\n  DIC File Columns:")
    for col in dic_info.get('columns', []):
        print(f"    - {col}")


def plot_force_vs_time(exp_num: int, config: ExperimentConfig,
                       force_info: Dict, output_dir: Path):
    """Create Force vs Time visualization."""

    if force_info.get('time_data') is None or force_info.get('force_data') is None:
        print("  [WARNING] Cannot create plot - missing data")
        return

    time = force_info['time_data']
    force = force_info['force_data']
    force_alt = force_info.get('force_data_alt')

    # Ensure arrays have same length
    min_len = min(len(time), len(force))
    time = time[:min_len]
    force = force[:min_len]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Full time series
    ax1 = axes[0]
    ax1.plot(time, force, 'b-', linewidth=0.8, label=f'Force (Col {config.force_column})', alpha=0.8)

    if force_alt is not None:
        force_alt = force_alt[:min_len]
        ax1.plot(time, force_alt, 'r--', linewidth=0.6, label=f'Force Alt (Col {config.force_column_alt})', alpha=0.6)

    ax1.set_xlabel('Time [s]', fontsize=12)
    ax1.set_ylabel('Force [kN]', fontsize=12)
    ax1.set_title(f'Experiment {exp_num}: {config.description}\nForce vs Time (Full)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f'Duration: {(time[-1]-time[0])/60:.1f} min\n'
    stats_text += f'Force Range: {force.min():.4f} - {force.max():.4f} kN\n'
    stats_text += f'Mean Force: {force.mean():.4f} kN\n'
    stats_text += f'Data Points: {len(time)}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot 2: First 100 seconds (zoom)
    ax2 = axes[1]
    mask = time <= 100
    if mask.sum() > 10:
        ax2.plot(time[mask], force[mask], 'b-', linewidth=1.0, label='Force', marker='o', markersize=2)
        if force_alt is not None:
            ax2.plot(time[mask], force_alt[mask], 'r--', linewidth=0.8, label='Force Alt', alpha=0.7)
        ax2.set_xlim(0, 100)
    else:
        ax2.plot(time[:100], force[:100], 'b-', linewidth=1.0, marker='o', markersize=2)

    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Force [kN]', fontsize=12)
    ax2.set_title('Force vs Time (First 100 seconds - Zoom)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / f'exp{exp_num}_force_vs_time.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n  [OK] Force vs Time plot saved: {output_file}")

    plt.show()
    plt.close()


def plot_dic_point_cloud(exp_num: int, config: ExperimentConfig,
                         dic_info: Dict, output_dir: Path):
    """Create DIC point cloud visualization."""

    df = dic_info.get('first_file_data')
    if df is None:
        print("  [WARNING] Cannot create DIC plot - missing data")
        return

    if 'x' not in df.columns or 'y' not in df.columns:
        print("  [WARNING] Cannot create DIC plot - x/y columns not found")
        return

    x = df['x'].values
    y = df['y'].values

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(x, y, c='steelblue', s=3, alpha=0.6, label='DIC Points')

    # Highlight boundaries
    ax.axhline(y.min(), color='red', linestyle='--', linewidth=1, alpha=0.7, label='Bottom')
    ax.axhline(y.max(), color='green', linestyle='--', linewidth=1, alpha=0.7, label='Top')
    ax.axvline(x.min(), color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Left')
    ax.axvline(x.max(), color='purple', linestyle='--', linewidth=1, alpha=0.7, label='Right')

    ax.set_xlabel('X [mm]', fontsize=12)
    ax.set_ylabel('Y [mm]', fontsize=12)
    ax.set_title(f'Experiment {exp_num}: DIC Point Cloud (t=0)\n{dic_info["n_nodes_first"]:,} nodes',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Add dimension annotations
    width = x.max() - x.min()
    height = y.max() - y.min()
    stats_text = f'Width: {width:.2f} mm\nHeight: {height:.2f} mm\nNodes: {len(x):,}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save figure
    output_file = output_dir / f'exp{exp_num}_dic_point_cloud.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  [OK] DIC point cloud plot saved: {output_file}")

    plt.show()
    plt.close()


def analyze_experiment(exp_num: int, show_plots: bool = True) -> Dict:
    """
    Main function to analyze a single experiment.

    Args:
        exp_num: Experiment number (1, 2, 3, or 4)
        show_plots: Whether to display and save plots

    Returns:
        Dictionary with analysis results
    """
    if exp_num not in EXPERIMENT_CONFIGS:
        print(f"[ERROR] Invalid experiment number: {exp_num}")
        print(f"        Available experiments: {list(EXPERIMENT_CONFIGS.keys())}")
        return {}

    config = EXPERIMENT_CONFIGS[exp_num]
    base_path = get_base_path()
    exp_path = base_path / str(exp_num)

    print(f"\n{'='*80}")
    print(f"  ANALYZING EXPERIMENT {exp_num}")
    print(f"  Path: {exp_path}")
    print(f"{'='*80}")

    if not exp_path.exists():
        print(f"[ERROR] Experiment directory not found: {exp_path}")
        return {}

    # Analyze force file
    force_path = exp_path / config.force_file
    print(f"\n  [1/3] Analyzing force file: {config.force_file}")
    force_info = analyze_force_file(force_path, config)

    if force_info.get('error'):
        print(f"        [WARNING] Error reading force file: {force_info['error']}")
    else:
        print(f"        Found {force_info['n_rows']} data rows")

    # Analyze DIC folder
    dic_path = exp_path / config.dic_folder
    print(f"\n  [2/3] Analyzing DIC folder: {config.dic_folder}")
    dic_info = analyze_dic_folder(dic_path, config)

    if dic_info.get('error'):
        print(f"        [WARNING] Error reading DIC data: {dic_info['error']}")
    else:
        print(f"        Found {dic_info['n_files']} DIC files, {dic_info['n_nodes_first']} nodes at t=0")

    # Print summary table
    print_summary_table(exp_num, config, force_info, dic_info)

    # Create output directory
    output_dir = Path(__file__).parent / "results_analysis"
    output_dir.mkdir(exist_ok=True)

    # Generate plots
    if show_plots:
        print(f"\n  [3/3] Generating visualizations...")
        plot_force_vs_time(exp_num, config, force_info, output_dir)
        plot_dic_point_cloud(exp_num, config, dic_info, output_dir)

    return {
        "experiment": exp_num,
        "config": config,
        "force_info": force_info,
        "dic_info": dic_info,
    }


def analyze_all_experiments(show_plots: bool = False) -> List[Dict]:
    """Analyze all experiments and create comparison table."""
    results = []

    for exp_num in EXPERIMENT_CONFIGS.keys():
        try:
            result = analyze_experiment(exp_num, show_plots=show_plots)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed to analyze experiment {exp_num}: {e}")

    # Print comparison table
    print("\n" + "="*120)
    print("  COMPARISON TABLE - ALL EXPERIMENTS")
    print("="*120)

    headers = ["Parameter", "Exp 1", "Exp 2", "Exp 3", "Exp 4"]
    print(f"  {headers[0]:<25} | {headers[1]:<20} | {headers[2]:<20} | {headers[3]:<20} | {headers[4]:<20}")
    print("  " + "-"*115)

    # Build comparison rows
    rows = [
        ("DIC Folder", [r.get('dic_info', {}).get('folder_name', 'N/A') for r in results]),
        ("Force File", [r.get('config', ExperimentConfig('','','',0,0,None,0,'',0,'')).force_file[:20] for r in results]),
        ("Force Skip Rows", [str(r.get('config', ExperimentConfig('','','',0,0,None,0,'',0,'')).force_skip_rows) for r in results]),
        ("Force Column", [f"Col {r.get('config', ExperimentConfig('','','',0,0,None,0,'',0,'')).force_column}" for r in results]),
        ("Time Column", [f"Col {r.get('config', ExperimentConfig('','','',0,0,None,0,'',0,'')).time_column}" for r in results]),
        ("DIC Separator", [r.get('config', ExperimentConfig('','','',0,0,None,0,',',0,'')).dic_separator for r in results]),
        ("DIC Skip Rows", [str(r.get('config', ExperimentConfig('','','',0,0,None,0,'',0,'')).dic_skip_rows) for r in results]),
        ("DIC Nodes", [f"{r.get('dic_info', {}).get('n_nodes_first', 0):,}" for r in results]),
    ]

    for param, values in rows:
        print(f"  {param:<25} | {values[0] if len(values) > 0 else 'N/A':<20} | {values[1] if len(values) > 1 else 'N/A':<20} | {values[2] if len(values) > 2 else 'N/A':<20} | {values[3] if len(values) > 3 else 'N/A':<20}")

    print("="*120)

    return results


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            exp_num = int(sys.argv[1])
            if sys.argv[1].lower() == 'all':
                analyze_all_experiments(show_plots=True)
            else:
                analyze_experiment(exp_num, show_plots=True)
        except ValueError:
            if sys.argv[1].lower() == 'all':
                analyze_all_experiments(show_plots=True)
            else:
                print(f"[ERROR] Invalid argument: {sys.argv[1]}")
                print("Usage: python dic_data_analysis.py [experiment_number|all]")
                sys.exit(1)
    else:
        # Interactive mode
        print("\n" + "="*60)
        print("  DIC DATA ANALYSIS TOOL")
        print("="*60)
        print("\n  Available experiments:")
        for exp_num, config in EXPERIMENT_CONFIGS.items():
            print(f"    {exp_num}: {config.description}")
        print("    all: Analyze all experiments (comparison table)")

        user_input = input("\n  Enter experiment number (1-4) or 'all': ").strip()

        if user_input.lower() == 'all':
            analyze_all_experiments(show_plots=True)
        else:
            try:
                exp_num = int(user_input)
                analyze_experiment(exp_num, show_plots=True)
            except ValueError:
                print(f"[ERROR] Invalid input: {user_input}")
                sys.exit(1)
