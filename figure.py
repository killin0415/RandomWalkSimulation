#!/usr/bin/env python3
"""
figure.py — Random Walk Simulation 繪圖腳本

Reads CSV data from the simulation and generates publication-quality figures.

Usage:
    python3 figure.py --datadir data --figdir figures
"""

import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ─── Global style ───────────────────────────────────────
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.titlesize'] = 13
rcParams['axes.labelsize'] = 12
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 200
rcParams['savefig.bbox'] = 'tight'

DIMS = [1, 2, 3, 4]
STEPS = [100, 1000, 10000, 100000, 1000000]
COLORS = {100: '#2196F3', 1000: '#4CAF50', 10000: '#FF9800',
           100000: '#E91E63', 1000000: '#9C27B0'}


def load_csv(path):
    """Load a CSV file as a dict of numpy arrays."""
    if not os.path.exists(path):
        return None
    data = {}
    with open(path) as f:
        header = f.readline().strip().split(',')
        cols = [[] for _ in header]
        for line in f:
            parts = line.strip().split(',')
            for i, v in enumerate(parts[:len(header)]):
                try:
                    cols[i].append(float(v))
                except ValueError:
                    cols[i].append(v)
    for i, h in enumerate(header):
        try:
            data[h] = np.array(cols[i], dtype=float)
        except (ValueError, TypeError):
            data[h] = cols[i]
    return data


def manual_pdf(values, num_bins=50):
    """Compute probability density manually (no library PDF estimation)."""
    values = np.array(values, dtype=float)
    vmin, vmax = values.min(), values.max()
    if vmin == vmax:
        return np.array([vmin]), np.array([1.0])
    bin_width = (vmax - vmin) / num_bins
    bin_edges = np.linspace(vmin, vmax, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts = np.zeros(num_bins)
    for v in values:
        idx = int((v - vmin) / bin_width)
        if idx >= num_bins:
            idx = num_bins - 1
        counts[idx] += 1
    # normalize to probability density
    density = counts / (len(values) * bin_width)
    return bin_centers, density


# ═══════════════════════════════════════════════════════════
# Figure 1: L1 and L2 distance distributions
#   Style: 4 rows (D=1..4) × 2 cols (L1, L2) per step count
#   Histogram bars + smooth curve + mean line + stats
# ═══════════════════════════════════════════════════════════

def manual_histogram(values, num_bins=60):
    """Compute histogram counts manually. Returns bin_edges, bin_centers, counts."""
    values = np.array(values, dtype=float)
    vmin, vmax = values.min(), values.max()
    if vmin == vmax:
        return np.array([vmin, vmin + 1]), np.array([vmin + 0.5]), np.array([len(values)])
    bin_edges = np.linspace(vmin, vmax, num_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts = np.zeros(num_bins)
    for v in values:
        idx = int((v - vmin) / bin_width)
        if idx >= num_bins:
            idx = num_bins - 1
        counts[idx] += 1
    return bin_edges, bin_centers, counts


def manual_smooth_curve(bin_centers, counts, sigma_bins=2.0):
    """Smooth histogram counts with a manual Gaussian kernel (no library KDE).
    sigma_bins controls smoothing width in units of bin spacing."""
    n_pts = len(bin_centers)
    if n_pts == 0:
        return bin_centers, counts
    # create a finer x grid for a smooth curve
    x_fine = np.linspace(bin_centers[0], bin_centers[-1], n_pts * 4)
    y_fine = np.zeros_like(x_fine)
    bin_width = bin_centers[1] - bin_centers[0] if n_pts > 1 else 1.0
    sigma = sigma_bins * bin_width

    for i in range(n_pts):
        if counts[i] == 0:
            continue
        # Gaussian kernel centered at bin_centers[i]
        w = np.exp(-0.5 * ((x_fine - bin_centers[i]) / sigma) ** 2)
        y_fine += counts[i] * w

    # Normalize so area under smooth curve ≈ area under histogram
    if y_fine.sum() > 0:
        hist_area = counts.sum() * bin_width
        smooth_area = np.trapz(y_fine, x_fine)
        if smooth_area > 0:
            y_fine *= hist_area / smooth_area
    return x_fine, y_fine


def plot_distance_distributions(datadir, figdir):
    for n in STEPS:
        # Check if any data exists for this n
        has_data = False
        for D in DIMS:
            if os.path.exists(os.path.join(datadir, f'dist_D{D}_n{n}.csv')):
                has_data = True
                break
        if not has_data:
            continue

        fig, axes = plt.subplots(4, 2, figsize=(14, 20))
        fig.suptitle(f'Random Walk Analysis (Steps={n}, Trials=1000)',
                     fontsize=18, fontweight='bold', y=0.995)

        for row, D in enumerate(DIMS):
            data = load_csv(os.path.join(datadir, f'dist_D{D}_n{n}.csv'))

            for col, (metric, color_hist, color_line, color_fill) in enumerate([
                ('l1', '#6BAED6', '#2171B5', '#C6DBEF'),   # blue tones
                ('l2', '#FC9272', '#CB181D', '#FCBBA1'),    # red/pink tones
            ]):
                ax = axes[row][col]
                label = 'L1' if col == 0 else 'L2'

                if data is None:
                    ax.set_title(f'Dim={D}: Distribution of {label}', fontsize=12)
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                            ha='center', va='center', fontsize=14, color='gray')
                    continue

                values = data[metric]
                num_walks = len(values)
                mean_val = values.mean()
                std_val = values.std()

                # Manual histogram
                num_bins = min(80, max(20, int(np.sqrt(num_walks) * 1.5)))
                bin_edges, bin_centers, counts = manual_histogram(values, num_bins=num_bins)
                bin_width = bin_edges[1] - bin_edges[0]

                # Draw histogram bars
                ax.bar(bin_centers, counts, width=bin_width * 0.9,
                       color=color_fill, edgecolor=color_hist,
                       linewidth=0.5, alpha=0.7, zorder=2)

                # Smooth curve overlay (manual Gaussian smoothing)
                x_smooth, y_smooth = manual_smooth_curve(bin_centers, counts, sigma_bins=1.8)
                ax.plot(x_smooth, y_smooth, color=color_line, linewidth=2.0, zorder=3)

                # Mean line (red dashed)
                ax.axvline(mean_val, color='#D32F2F', linestyle='--', linewidth=1.8, zorder=4)

                # Annotations
                ax.set_title(f'Dim={D}: Distribution of {label}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Value', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)

                # Std and Trials in top-right
                ax.text(0.98, 0.95, f'Std: {std_val:.2f}\nTrials: {num_walks}',
                        transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='gray', alpha=0.8))

                # Mean annotation in legend style
                ax.plot([], [], color='#D32F2F', linestyle='--', linewidth=1.8,
                        label=f'Mean: {mean_val:.2f}')
                ax.legend(loc='lower right', fontsize=9, framealpha=0.8)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(os.path.join(figdir, f'distance_n{n}.pdf'))
        fig.savefig(os.path.join(figdir, f'distance_n{n}.png'))
        plt.close(fig)
        print(f'  [OK] distance_n{n}.pdf')


# ═══════════════════════════════════════════════════════════
# Figure 2: Section (quadrant) occupancy verification
# ═══════════════════════════════════════════════════════════
def plot_section_occupancy(datadir, figdir):
    for D in DIMS:
        num_sec = 2 ** D
        # pick the largest available n
        for n in reversed(STEPS):
            data = load_csv(os.path.join(datadir, f'section_D{D}_n{n}.csv'))
            if data is not None:
                break
        else:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        sec_keys = [f'sec{s}' for s in range(num_sec)]
        means = [data[k].mean() for k in sec_keys]
        total = sum(means)
        fracs = [m / total for m in means]
        expected = 1.0 / num_sec

        bars = ax.bar(range(num_sec), fracs, color='#42A5F5', edgecolor='#1565C0', linewidth=1.2)
        ax.axhline(expected, color='#E53935', linestyle='--', linewidth=1.5,
                   label=f'Expected = 1/{num_sec} = {expected:.4f}')
        ax.set_xlabel('Section Index')
        ax.set_ylabel('Fraction of Time Steps')
        ax.set_title(f'Section Occupancy (D={D}, n={n})', fontweight='bold')
        ax.set_xticks(range(num_sec))
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig.savefig(os.path.join(figdir, f'section_D{D}.pdf'))
        fig.savefig(os.path.join(figdir, f'section_D{D}.png'))
        plt.close(fig)
        print(f'  [OK] section_D{D}.pdf')


# ═══════════════════════════════════════════════════════════
# Figure 3: Return-to-origin distribution
# ═══════════════════════════════════════════════════════════
def plot_return_to_origin(datadir, figdir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('First Return to Origin Distribution', fontsize=14, fontweight='bold')

    for idx, D in enumerate(DIMS):
        ax = axes[idx // 2][idx % 2]
        all_first_returns = []
        never_returned_counts = {}

        for n in STEPS:
            path = os.path.join(datadir, f'return_D{D}_n{n}.csv')
            data = load_csv(path)
            if data is None:
                continue

            first_returns = data['first_return']
            valid = first_returns[first_returns > 0]
            never_count = np.sum(first_returns < 0)
            never_returned_counts[n] = (never_count, len(first_returns))

            if len(valid) > 0:
                all_first_returns.extend(valid.tolist())

        if len(all_first_returns) > 0:
            all_first_returns = np.array(all_first_returns)
            centers, density = manual_pdf(all_first_returns,
                                          num_bins=min(50, max(10, int(len(all_first_returns) / 20))))
            ax.plot(centers, density, color='#1976D2', linewidth=1.5)
            ax.fill_between(centers, density, alpha=0.2, color='#1976D2')

        # annotate never-returned fractions
        text_lines = []
        for n in STEPS:
            if n in never_returned_counts:
                nc, total = never_returned_counts[n]
                pct = 100.0 * nc / total
                text_lines.append(f'n={n}: {pct:.1f}% never returned')
        if text_lines:
            ax.text(0.98, 0.98, '\n'.join(text_lines), transform=ax.transAxes,
                    fontsize=7, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('First Return Step')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'D = {D}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(figdir, 'return_to_origin.pdf'))
    fig.savefig(os.path.join(figdir, 'return_to_origin.png'))
    plt.close(fig)
    print('  [OK] return_to_origin.pdf')


# ═══════════════════════════════════════════════════════════
# Figure 4: Expected return steps summary
# ═══════════════════════════════════════════════════════════
def plot_expected_return(datadir, figdir):
    fig, ax = plt.subplots(figsize=(10, 6))

    for D in DIMS:
        means = []
        ns_used = []
        for n in STEPS:
            data = load_csv(os.path.join(datadir, f'return_D{D}_n{n}.csv'))
            if data is None:
                continue
            first_returns = data['first_return']
            valid = first_returns[first_returns > 0]
            if len(valid) > 5:
                means.append(valid.mean())
                ns_used.append(n)
        if means:
            ax.plot(ns_used, means, 'o-', label=f'D={D}', linewidth=2, markersize=6)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Walk Length n')
    ax.set_ylabel('Mean First Return Step')
    ax.set_title('Expected First Return to Origin vs Walk Length', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    fig.savefig(os.path.join(figdir, 'expected_return.pdf'))
    fig.savefig(os.path.join(figdir, 'expected_return.png'))
    plt.close(fig)
    print('  [OK] expected_return.pdf')


# ═══════════════════════════════════════════════════════════
# Figure 5: Number of returns to origin
# ═══════════════════════════════════════════════════════════
def plot_num_returns(datadir, figdir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Number of Returns to Origin', fontsize=14, fontweight='bold')

    for idx, D in enumerate(DIMS):
        ax = axes[idx // 2][idx % 2]
        for n in STEPS:
            data = load_csv(os.path.join(datadir, f'return_D{D}_n{n}.csv'))
            if data is None:
                continue
            num_returns = data['num_returns']
            centers, density = manual_pdf(num_returns,
                                          num_bins=min(40, max(5, int(num_returns.max()) + 1)))
            ax.plot(centers, density, label=f'n={n}', color=COLORS[n], linewidth=1.5)
        ax.set_xlabel('Number of Returns')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'D = {D}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(figdir, 'num_returns.pdf'))
    fig.savefig(os.path.join(figdir, 'num_returns.png'))
    plt.close(fig)
    print('  [OK] num_returns.pdf')


# ═══════════════════════════════════════════════════════════
# Figure 6: m/n distribution (1D only)
# ═══════════════════════════════════════════════════════════
def plot_m_over_n(datadir, figdir):
    fig, ax = plt.subplots(figsize=(10, 6))

    for n in STEPS:
        data = load_csv(os.path.join(datadir, f'onedim_n{n}.csv'))
        if data is None:
            continue
        m_over_n = data['m_over_n']
        centers, density = manual_pdf(m_over_n, num_bins=50)
        ax.plot(centers, density, label=f'n={n}', color=COLORS[n], linewidth=1.5)
        ax.fill_between(centers, density, alpha=0.1, color=COLORS[n])

    # theoretical arcsine distribution density: f(x) = 1/(π√(x(1-x)))
    x_theory = np.linspace(0.505, 0.995, 500)
    y_theory = 1.0 / (np.pi * np.sqrt(x_theory * (1.0 - x_theory)))
    ax.plot(x_theory, y_theory, 'k--', linewidth=2, label='Arcsine law (theory)')

    ax.set_xlabel('m / n')
    ax.set_ylabel('Probability Density')
    ax.set_title('Distribution of m/n (1D Random Walk — Arcsine Law)',
                 fontweight='bold')
    ax.set_xlim(0.5, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(figdir, 'm_over_n.pdf'))
    fig.savefig(os.path.join(figdir, 'm_over_n.png'))
    plt.close(fig)
    print('  [OK] m_over_n.pdf')


# ═══════════════════════════════════════════════════════════
# Figure 7: Distance scaling with sqrt(n)
# ═══════════════════════════════════════════════════════════
def plot_distance_scaling(datadir, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Distance Scaling: Mean Distance vs √n', fontsize=14, fontweight='bold')

    for D in DIMS:
        l1_means = []
        l2_means = []
        ns_used = []
        for n in STEPS:
            data = load_csv(os.path.join(datadir, f'dist_D{D}_n{n}.csv'))
            if data is None:
                continue
            l1_means.append(data['l1'].mean())
            l2_means.append(data['l2'].mean())
            ns_used.append(n)

        if ns_used:
            sqrt_n = [np.sqrt(x) for x in ns_used]
            axes[0].plot(sqrt_n, l1_means, 'o-', label=f'D={D}', linewidth=2, markersize=5)
            axes[1].plot(sqrt_n, l2_means, 'o-', label=f'D={D}', linewidth=2, markersize=5)

    for ax, title in zip(axes, ['L1-norm', 'L2-norm']):
        ax.set_xlabel('√n')
        ax.set_ylabel(f'Mean {title} Distance')
        ax.set_title(f'{title} Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(figdir, 'distance_scaling.pdf'))
    fig.savefig(os.path.join(figdir, 'distance_scaling.png'))
    plt.close(fig)
    print('  [OK] distance_scaling.pdf')


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Random Walk Figure Generator')
    parser.add_argument('--datadir', default='data', help='Input data directory')
    parser.add_argument('--figdir', default='figures', help='Output figure directory')
    args = parser.parse_args()

    os.makedirs(args.figdir, exist_ok=True)

    print('Generating figures...')
    print()

    print('[1/7] Distance distributions...')
    plot_distance_distributions(args.datadir, args.figdir)

    print('[2/7] Section occupancy...')
    plot_section_occupancy(args.datadir, args.figdir)

    print('[3/7] Return to origin distributions...')
    plot_return_to_origin(args.datadir, args.figdir)

    print('[4/7] Expected return steps...')
    plot_expected_return(args.datadir, args.figdir)

    print('[5/7] Number of returns...')
    plot_num_returns(args.datadir, args.figdir)

    print('[6/7] m/n distribution (1D)...')
    plot_m_over_n(args.datadir, args.figdir)

    print('[7/7] Distance scaling...')
    plot_distance_scaling(args.datadir, args.figdir)

    print()
    print(f'All figures saved to {args.figdir}/')


if __name__ == '__main__':
    main()
