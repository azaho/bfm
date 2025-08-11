#!/usr/bin/env python3
"""
Script to visualize attention scores on brain hemispheres.
Creates PDFs with brain plots showing CLS attention to electrodes for all layers and subjects.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import argparse
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
import torch
import pandas as pd
from evaluation.neuroprobe.config import *
from subject.braintreebank import BrainTreebankSubject as Subject
import matplotlib.font_manager as fm

# Set up fonts
try:
    font_path = 'assets/font_arial.ttf'
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Arial'
except:
    print("Warning: Could not load Arial font, using default")

# Brain image setup
matlab_xlim = (-108.0278, 108.0278)
matlab_ylim = (-72.9774, 72.9774)

base_path = os.path.join(ROOT_DIR, 'localization')
left_hem_file_name = 'left_hem_clean.png'
right_hem_file_name = 'right_hem_clean.png'
coords_file_name = 'elec_coords_full.csv'

# Load brain images and coordinates
left_hem_img = plt.imread(os.path.join(base_path, left_hem_file_name))
right_hem_img = plt.imread(os.path.join(base_path, right_hem_file_name))
coords_df = pd.read_csv(os.path.join(base_path, coords_file_name))
split_elec_id = coords_df['ID'].str.split('-')
coords_df['Subject'] = [t[0] for t in split_elec_id]
coords_df['Electrode'] = [t[1] for t in split_elec_id]

def scale(x, s, d):
    """Scale Matlab electrode locations to Python format"""
    return -(x - d) * s

# Scale coordinates
x_scale = left_hem_img.shape[1] / (matlab_xlim[1] - matlab_xlim[0])
y_scale_l = left_hem_img.shape[0] / (matlab_ylim[1] - matlab_ylim[0])
y_scale_r = right_hem_img.shape[0] / (matlab_ylim[1] - matlab_ylim[0])

scaled_coords_df = coords_df.copy()

# Scale left hemisphere coordinates
scaled_coords_df.loc[scaled_coords_df['Hemisphere'] == 1, 'X'] = coords_df.loc[coords_df['Hemisphere'] == 1, 'X'].apply(lambda x: scale(x, x_scale, matlab_xlim[1]))
scaled_coords_df.loc[scaled_coords_df['Hemisphere'] == 1, 'Y'] = coords_df.loc[coords_df['Hemisphere'] == 1, 'Y'].apply(lambda x: scale(x, y_scale_l, matlab_ylim[1]))

# Scale right hemisphere coordinates
scaled_coords_df.loc[scaled_coords_df['Hemisphere'] == 0, 'X'] = coords_df.loc[coords_df['Hemisphere'] == 0, 'X'].apply(lambda x: -scale(x, y_scale_r, matlab_xlim[0]))
scaled_coords_df.loc[scaled_coords_df['Hemisphere'] == 0, 'Y'] = coords_df.loc[coords_df['Hemisphere'] == 0, 'Y'].apply(lambda x: scale(x, y_scale_r, matlab_ylim[1]))

def plot_hemisphere_axis(electrodes, colors=None, sizes=None, ax=None, hemisphere="left", title=None, vmin=0, vmax=0.1):
    """
    Plot electrodes on a single hemisphere
    
    Args:
        electrodes: dict of format {<subject>: [<electrode>]}
        colors: dict mapping electrode labels to colors
        sizes: dict mapping electrode labels to sizes (for attention scores)
        ax: matplotlib axis
        hemisphere: "left" or "right"
        title: optional title
        vmin, vmax: color scale limits
    """
    if colors is None: 
        colors = {e: 'white' for e in electrodes}
    if sizes is None:
        sizes = {e: 200 for e in electrodes}
    
    ax.set_aspect('equal')

    if hemisphere=="left":
        ax.imshow(left_hem_img)
    elif hemisphere=="right":
        ax.imshow(right_hem_img)
    
    ax.axis('off')
    assert hemisphere in ["left", "right"]
    hem_index = 1 if hemisphere=="left" else 0

    plot_title = f'{hemisphere} hemisphere'
    if title: plot_title += f' {title}'
   
    all_x, all_y, all_colors, all_sizes = [], [], [], []
    for s in electrodes:
        for e in electrodes[s]:
            x = list(scaled_coords_df[(scaled_coords_df.Subject == s) & (scaled_coords_df.Electrode == e) & (scaled_coords_df.Hemisphere==hem_index)]['X'])
            y = list(scaled_coords_df[(scaled_coords_df.Subject == s) & (scaled_coords_df.Electrode == e) & (scaled_coords_df.Hemisphere==hem_index)]['Y'])
            if (len(x) == 0 or len(y) == 0) and hemisphere == "left":
                print(f"Warning: No coordinates found for subject {s} and electrode {e} in {hemisphere} hemisphere")
            if len(x) == 0 or len(y) == 0:
                continue
            
            assert len(x) == len(y) == 1, f"For subject {s} and electrode {e}, x: {x}, y: {y}"

            all_x += x
            all_y += y
            all_colors += [colors.get(e, 'white')]
            all_sizes += [sizes.get(e, 200)]
    
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_colors = np.array(all_colors)
    all_sizes = np.array(all_sizes)
    
    # Sort points by size so larger values appear on top
    sort_idx = np.argsort(all_sizes)
    sc = ax.scatter(all_x[sort_idx], all_y[sort_idx], c=all_colors[sort_idx], s=200, 
                   vmin=vmin, vmax=vmax, edgecolors='black', cmap='viridis', alpha=0.8)
    
    return sc

def plot_hemispheres_separately(electrodes, ax1, ax2, colors=None, sizes=None, vmin=0, vmax=0.1):
    """Plot both hemispheres side by side"""
    sc1 = plot_hemisphere_axis(electrodes, colors=colors, sizes=sizes, ax=ax1, hemisphere="right", vmin=vmin, vmax=vmax)
    sc2 = plot_hemisphere_axis(electrodes, colors=colors, sizes=sizes, ax=ax2, hemisphere="left", vmin=vmin, vmax=vmax)
    return sc1, sc2

def create_attention_brain_plots(attention_file_path, output_pdf_path):
    """
    Create brain plots for attention scores from a single attention file
    
    Args:
        attention_file_path: path to the .npy attention file
        output_pdf_path: path for the output PDF
    """
    print(f"Loading attention matrices from {attention_file_path}")
    
    # Load attention matrices
    attention_matrices = np.load(attention_file_path, allow_pickle=True)
    x = np.array(attention_matrices.item()['analysis_results']).mean(axis=0)
    electrode_labels = attention_matrices.item()['electrode_labels']
    
    # Extract subject ID from file path
    filename = os.path.basename(attention_file_path)
    subject_id = int(filename.split('btbank')[1].split('_')[0])
    
    print(f"Processing subject {subject_id} with {len(x)} layers and {len(electrode_labels)-1} electrodes")
    
    with PdfPages(output_pdf_path) as pdf:
        # For each layer
        for layer_idx in range(len(x)):
            # Get CLS attention to electrodes (row 0, columns 1:)
            cls_attention = x[layer_idx][0, 1:]
            
            # Create colors and sizes based on attention scores
            colors = {}
            sizes = {}
            
            # Normalize attention scores for color mapping
            norm = Normalize(cls_attention.min(), cls_attention.max())
            cmap = plt.cm.viridis
            
            for i, electrode_label in enumerate(electrode_labels[1:]):  # Skip CLS
                attention_score = cls_attention[i]
                
                # Color based on attention score (use normalized value for color)
                color = cmap(norm(attention_score))
                colors[electrode_label] = attention_score  # Use raw score for color mapping
                
                # Use constant size for all electrodes
                size = 200  # Fixed size for all electrodes
                sizes[electrode_label] = size
            
            # Create brain plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 9), gridspec_kw={'wspace': 0.1})
            
            # Prepare electrode data for plotting
            attention_matrix_electrodes = {f'sub_{subject_id}': electrode_labels[1:]}
            
            # Plot hemispheres
            sc1, sc2 = plot_hemispheres_separately(
                attention_matrix_electrodes, 
                axes[0], 
                axes[1], 
                colors=colors, 
                sizes=sizes,
                vmin=cls_attention.min(),
                vmax=cls_attention.max()
            )
            
            # Add colorbar
            cbar = plt.colorbar(sc1, ax=axes, shrink=0.8, aspect=20)
            cbar.set_label('CLS Attention Score', fontsize=12)
            
            # Add title
            fig.suptitle(f'Subject {subject_id} - Layer {layer_idx} - CLS Attention to Electrodes', fontsize=16)
            axes[0].set_title('Right Hemisphere', fontsize=12)
            axes[1].set_title('Left Hemisphere', fontsize=12)
            
            # Save to PDF
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            print(f"  Saved brain plot for layer {layer_idx}")

def create_attention_matrix_plots(attention_file_path, output_pdf_path):
    """
    Create traditional matrix heatmap plots for attention scores
    
    Args:
        attention_file_path: path to the .npy attention file
        output_pdf_path: path for the output PDF
    """
    print(f"Loading attention matrices from {attention_file_path}")
    
    # Load attention matrices
    attention_matrices = np.load(attention_file_path, allow_pickle=True)
    x = np.array(attention_matrices.item()['analysis_results']).mean(axis=0)
    electrode_labels = attention_matrices.item()['electrode_labels']
    
    # Extract subject ID from file path
    filename = os.path.basename(attention_file_path)
    subject_id = int(filename.split('btbank')[1].split('_')[0])
    
    print(f"Processing subject {subject_id} with {len(x)} layers and {len(electrode_labels)-1} electrodes")
    
    with PdfPages(output_pdf_path) as pdf:
        # For each layer
        for layer_idx in range(len(x)):
            # Full attention matrix with CLS
            plt.figure(figsize=(25, 25))
            sns.heatmap(x[layer_idx], cmap='viridis', annot=False, fmt='.3f', square=True,
                        xticklabels=electrode_labels, yticklabels=electrode_labels)
            plt.title(f'Subject {subject_id} - Layer {layer_idx} - Full Attention Matrix (with CLS)', fontsize=16)
            pdf.savefig(bbox_inches='tight', dpi=300)
            plt.close()
            
            # Attention matrix without CLS
            plt.figure(figsize=(25, 25))
            sns.heatmap(x[layer_idx][1:,1:], cmap='viridis', annot=False, fmt='.3f', square=True,
                        xticklabels=electrode_labels[1:], yticklabels=electrode_labels[1:])
            plt.title(f'Subject {subject_id} - Layer {layer_idx} - Electrode-to-Electrode Attention (no CLS)', fontsize=16)
            pdf.savefig(bbox_inches='tight', dpi=300)
            plt.close()
            
            # CLS attention to electrodes (row 0, columns 1:)
            plt.figure(figsize=(25, 1))
            sns.heatmap(x[layer_idx][0:1, 1:], cmap='viridis', annot=False, fmt='.3f', square=False,
                        xticklabels=electrode_labels[1:], yticklabels=False)
            plt.title(f'Subject {subject_id} - Layer {layer_idx} - CLS Attention to Electrodes', fontsize=16)
            pdf.savefig(bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"  Saved matrix plots for layer {layer_idx}")

def create_combined_plots(attention_file_path, output_pdf_path):
    """
    Create both brain plots and matrix plots in a single PDF
    
    Args:
        attention_file_path: path to the .npy attention file
        output_pdf_path: path for the output PDF
    """
    print(f"Loading attention matrices from {attention_file_path}")
    
    # Load attention matrices
    attention_matrices = np.load(attention_file_path, allow_pickle=True)
    x = np.array(attention_matrices.item()['analysis_results']).mean(axis=0)
    electrode_labels = attention_matrices.item()['electrode_labels']
    
    # Extract subject ID from file path
    filename = os.path.basename(attention_file_path)
    subject_id = int(filename.split('btbank')[1].split('_')[0])
    
    print(f"Processing subject {subject_id} with {len(x)} layers and {len(electrode_labels)-1} electrodes")
    
    with PdfPages(output_pdf_path) as pdf:
        # For each layer
        for layer_idx in range(len(x)):
            # Get CLS attention to electrodes (row 0, columns 1:)
            cls_attention = x[layer_idx][0, 1:]
            
            # Create colors and sizes based on attention scores
            colors = {}
            sizes = {}
            
            # Normalize attention scores for color mapping
            norm = Normalize(cls_attention.min(), cls_attention.max())
            cmap = plt.cm.viridis
            
            for i, electrode_label in enumerate(electrode_labels[1:]):  # Skip CLS
                attention_score = cls_attention[i]
                
                # Color based on attention score (use normalized value for color)
                color = cmap(norm(attention_score))
                colors[electrode_label] = attention_score  # Use raw score for color mapping
                
                # Size based on attention score using min-max scaling
                # This ensures we use the full range of sizes from min to max
                min_score = cls_attention.min()
                max_score = cls_attention.max()
                normalized_score = (attention_score - min_score) / (max_score - min_score)
                size = 50 + normalized_score * 500  # Range from 50 to 550
                sizes[electrode_label] = size
            
            # 1. Brain plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 9), gridspec_kw={'wspace': 0.1})
            
            # Prepare electrode data for plotting
            attention_matrix_electrodes = {f'sub_{subject_id}': electrode_labels[1:]}
            
            # Plot hemispheres
            sc1, sc2 = plot_hemispheres_separately(
                attention_matrix_electrodes, 
                axes[0], 
                axes[1], 
                colors=colors, 
                sizes=sizes,
                vmin=cls_attention.min(),
                vmax=cls_attention.max()
            )
            
            # Add colorbar
            cbar = plt.colorbar(sc1, ax=axes, shrink=0.8, aspect=20)
            cbar.set_label('CLS Attention Score', fontsize=12)
            
            # Add title
            fig.suptitle(f'Subject {subject_id} - Layer {layer_idx} - CLS Attention to Electrodes (Brain View)', fontsize=16)
            axes[0].set_title('Right Hemisphere', fontsize=12)
            axes[1].set_title('Left Hemisphere', fontsize=12)
            
            # Save to PDF
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            # 2. Full attention matrix with CLS
            plt.figure(figsize=(25, 25))
            sns.heatmap(x[layer_idx], cmap='viridis', annot=False, fmt='.3f', square=True,
                        xticklabels=electrode_labels, yticklabels=electrode_labels)
            plt.title(f'Subject {subject_id} - Layer {layer_idx} - Full Attention Matrix (with CLS)', fontsize=16)
            pdf.savefig(bbox_inches='tight', dpi=300)
            plt.close()
            
            # 3. Attention matrix without CLS
            plt.figure(figsize=(25, 25))
            sns.heatmap(x[layer_idx][1:,1:], cmap='viridis', annot=False, fmt='.3f', square=True,
                        xticklabels=electrode_labels[1:], yticklabels=electrode_labels[1:])
            plt.title(f'Subject {subject_id} - Layer {layer_idx} - Electrode-to-Electrode Attention (no CLS)', fontsize=16)
            pdf.savefig(bbox_inches='tight', dpi=300)
            plt.close()
            
            # 4. CLS attention to electrodes (row 0, columns 1:)
            plt.figure(figsize=(25, 1))
            sns.heatmap(x[layer_idx][0:1, 1:], cmap='viridis', annot=False, fmt='.3f', square=False,
                        xticklabels=electrode_labels[1:], yticklabels=False)
            plt.title(f'Subject {subject_id} - Layer {layer_idx} - CLS Attention to Electrodes (Matrix View)', fontsize=16)
            pdf.savefig(bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"  Saved all plots for layer {layer_idx}")

def process_all_subjects(base_path, output_dir, plot_type='combined'):
    """
    Process all subjects and create individual PDFs for each
    
    Args:
        base_path: directory containing attention files
        output_dir: directory to save output PDFs
        plot_type: 'brain', 'matrix', or 'combined'
    """
    print(f"Searching for attention files in {base_path}")
    
    # Find all attention files
    attention_files = glob.glob(os.path.join(base_path, "**/key_electrodes_btbank*_attention.npy"), recursive=True)
    
    if not attention_files:
        print(f"No attention files found in {base_path}")
        return
    
    print(f"Found {len(attention_files)} attention files")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for attention_file in attention_files:
        # Extract subject and trial info from filename
        filename = os.path.basename(attention_file)
        parts = filename.replace('.npy', '').split('_')
        subject_id = parts[2].replace('btbank', '')
        trial_id = parts[3]
        eval_name = parts[4]
        
        # Create output PDF path
        output_pdf = os.path.join(output_dir, f'btbank{subject_id}_{trial_id}_attention_matrices.pdf')
        
        print(f"\nProcessing {filename}...")
        
        if plot_type == 'brain':
            create_attention_brain_plots(attention_file, output_pdf)
        elif plot_type == 'matrix':
            create_attention_matrix_plots(attention_file, output_pdf)
        else:  # combined
            create_combined_plots(attention_file, output_pdf)
            
        print(f"Created {output_pdf}")

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description='Create brain plots for attention scores')
    parser.add_argument('--base_path', type=str, 
                       default="runs/data/andrii0_wd0.0001_dr0.1_rTEMP/key_electrodes/model_epoch100/",
                       help='Directory containing attention files')
    parser.add_argument('--output_dir', type=str, 
                       default="analyses/roshnipm/attention_matrices/andrii0",
                       help='Directory to save output PDFs')
    parser.add_argument('--single_file', type=str, 
                       help='Process only a single attention file')
    parser.add_argument('--plot_type', type=str, choices=['brain', 'matrix', 'combined'],
                       default='combined',
                       help='Type of plots to generate: brain plots, matrix plots, or both')
    
    args = parser.parse_args()
    
    if args.single_file:
        # Process single file
        if not os.path.exists(args.single_file):
            print(f"Error: File {args.single_file} not found")
            return
        
        filename = os.path.basename(args.single_file)
        output_pdf = os.path.join(args.output_dir, f'attention_brain_plots_{filename.replace(".npy", ".pdf")}')
        os.makedirs(args.output_dir, exist_ok=True)
        
        print(f"Processing single file: {args.single_file}")
        
        if args.plot_type == 'brain':
            create_attention_brain_plots(args.single_file, output_pdf)
        elif args.plot_type == 'matrix':
            create_attention_matrix_plots(args.single_file, output_pdf)
        else:  # combined
            create_combined_plots(args.single_file, output_pdf)
            
        print(f"Created {output_pdf}")
    else:
        # Process all subjects
        process_all_subjects(args.base_path, args.output_dir, args.plot_type)
        print(f"\nAll processing complete! PDFs saved to {args.output_dir}")

if __name__ == "__main__":
    main()

