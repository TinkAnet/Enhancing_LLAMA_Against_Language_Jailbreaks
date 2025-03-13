import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from scipy import stats

# Create output directories
os.makedirs("graph_2", exist_ok=True)
os.makedirs("graph_2/harm_scores", exist_ok=True)
os.makedirs("graph_2/prompt_scores", exist_ok=True)
os.makedirs("graph_2/relationships", exist_ok=True)
os.makedirs("graph_2/reasoning_comparison", exist_ok=True)

# Path to results directory
base_dir = "explicit_judgment_result_2"
reasoning_dir = "Explicit_judgment_result_reasoning"
categories = ["fraud", "politics", "pornography_sexual_minors", "race", "religion", "suicide", "terrorism", "violence"]

# Dictionary to store data from all categories
all_data = {}
reasoning_data = {}

# Color definitions for consistent styling
COLORS = {
    'harm': '#FF5733',  # Red-orange for harm
    'prompt': '#33A2FF',  # Blue for prompt
    'relationship': '#8A2BE2',  # Purple for relationships
    'reasoning': '#66BB6A',  # Green for reasoning data
    'highlight': '#FFC107',  # Amber for highlights
    'category_colors': plt.cm.tab10,  # Color map for categories
    'correlation_positive': '#00C853',  # Green for positive correlation
    'correlation_negative': '#D50000',  # Red for negative correlation
    'correlation_neutral': '#757575',  # Gray for neutral correlation
}

# Function to extract scores from a results.json file
def extract_scores(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        harm_scores = []
        prompt_scores = []
        
        # Iterate through each item in the JSON data
        for item in data:
            try:
                # Convert scores to float, handle potential string values
                harm_score = float(item.get('Harm_Score', 0))
                prompt_score = float(item.get('Prompt_Score', 0))
                
                harm_scores.append(harm_score)
                prompt_scores.append(prompt_score)
            except (ValueError, TypeError):
                continue
                
        return harm_scores, prompt_scores
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return [], []

# Function to load reasoning data from Explicit_judgment_result_reasoning folder
def load_reasoning_data(category):
    file_path = os.path.join(reasoning_dir, f"reasoning_1_{category}_SAP200.json")
    print(f"Looking for reasoning file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Warning: Reasoning file not found at {file_path}")
        return []
        
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        print(f"Successfully loaded reasoning data for {category}. Found {len(data)} entries.")
        
        # Extract Final Severity Score from reasoning file
        reasoning_scores = []
        
        # Debug: Print the first entry to see its structure
        if len(data) > 0:
            print(f"First reasoning entry structure for {category}: {list(data[0].keys())}")
        
        # The field we want is 'Final Severity Score'
        field_name = 'Final Severity Score'
        
        for item in data:
            if field_name in item:
                try:
                    # Convert the score to float
                    score_value = float(item[field_name])
                    reasoning_scores.append(score_value)
                except (ValueError, TypeError):
                    # If conversion fails, try to extract numeric part if it's a string
                    try:
                        score_str = str(item[field_name])
                        # Try to extract numeric part if it's in a format like "7.5/10"
                        if '/' in score_str:
                            numerator = score_str.split('/')[0]
                            score_value = float(numerator)
                            reasoning_scores.append(score_value)
                        else:
                            print(f"Warning: Unable to parse reasoning score: {score_str}")
                    except:
                        print(f"Warning: Unable to convert reasoning score to float: {item[field_name]}")
        
        print(f"Extracted {len(reasoning_scores)} reasoning scores for {category}")
        return reasoning_scores
    except Exception as e:
        print(f"Error reading reasoning file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

# Generate separate visualizations for harm scores
def visualize_harm_scores(category, harm_scores):
    print(f"Generating harm score visualizations for {category}...")
    
    # Set modern style
    plt.style.use('ggplot')
    
    # 1. Histogram with KDE
    plt.figure(figsize=(10, 6))
    sns.histplot(harm_scores, kde=True, color=COLORS['harm'], bins=20)
    plt.title(f'{category.capitalize()} - Harm Score Distribution', fontsize=16)
    plt.xlabel('Harm Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"graph_2/harm_scores/{category}_harm_hist.png", dpi=300)
    plt.close()
    
    # 2. Box plot with strip plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=harm_scores, color=COLORS['harm'], width=0.3)
    sns.stripplot(y=harm_scores, color='black', alpha=0.5, size=4)
    plt.title(f'{category.capitalize()} - Harm Score Box Plot', fontsize=16)
    plt.ylabel('Harm Score', fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"graph_2/harm_scores/{category}_harm_box.png", dpi=300)
    plt.close()
    
    # 3. Violin plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(y=harm_scores, color=COLORS['harm'], inner="quart")
    plt.title(f'{category.capitalize()} - Harm Score Violin Plot', fontsize=16)
    plt.ylabel('Harm Score', fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"graph_2/harm_scores/{category}_harm_violin.png", dpi=300)
    plt.close()
    
    # 4. ECDF (Empirical Cumulative Distribution Function)
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(harm_scores, color=COLORS['harm'], linewidth=2)
    plt.title(f'{category.capitalize()} - Harm Score Cumulative Distribution', fontsize=16)
    plt.xlabel('Harm Score', fontsize=14)
    plt.ylabel('Cumulative Probability', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"graph_2/harm_scores/{category}_harm_ecdf.png", dpi=300)
    plt.close()
    
    # 5. Ridgeline plot if we have enough data points and sufficient variance
    if len(harm_scores) >= 20:
        # Check for variance
        if np.var(harm_scores) > 0.001:  # Only create if there's meaningful variance
            fig, axes = plt.subplots(figsize=(10, 6))
            harm_scores_series = pd.Series(harm_scores, name='Harm Scores')
            for idx, frac in enumerate(np.linspace(0.1, 1.0, 5)):
                sample_size = max(int(len(harm_scores) * frac), 5)  # Ensure minimum sample size
                sample = np.random.choice(harm_scores, sample_size, replace=False)
                try:
                    sns.kdeplot(sample, bw_adjust=0.8, alpha=0.6, linewidth=2, 
                              color=COLORS['harm'], ax=axes, label=f'Sample {idx+1}',
                              warn_singular=False)  # Suppress zero variance warnings
                except:
                    pass  # Skip if KDE fails
            plt.title(f'{category.capitalize()} - Harm Score Density Variations', fontsize=16)
            plt.xlabel('Harm Score', fontsize=14)
            plt.ylabel('Density', fontsize=14)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"graph_2/harm_scores/{category}_harm_ridgeline.png", dpi=300)
            plt.close()
    else:
            print(f"Skipping ridgeline plot for {category} due to insufficient variance")

# Generate separate visualizations for prompt scores
def visualize_prompt_scores(category, prompt_scores):
    print(f"Generating prompt score visualizations for {category}...")
    
    # Set modern style
    plt.style.use('ggplot')
    
    # 1. Histogram with KDE
    plt.figure(figsize=(10, 6))
    sns.histplot(prompt_scores, kde=True, color=COLORS['prompt'], bins=20)
    plt.title(f'{category.capitalize()} - Prompt Score Distribution', fontsize=16)
    plt.xlabel('Prompt Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"graph_2/prompt_scores/{category}_prompt_hist.png", dpi=300)
    plt.close()
    
    # 2. Box plot with strip plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=prompt_scores, color=COLORS['prompt'], width=0.3)
    sns.stripplot(y=prompt_scores, color='black', alpha=0.5, size=4)
    plt.title(f'{category.capitalize()} - Prompt Score Box Plot', fontsize=16)
    plt.ylabel('Prompt Score', fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"graph_2/prompt_scores/{category}_prompt_box.png", dpi=300)
    plt.close()
    
    # 3. Violin plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(y=prompt_scores, color=COLORS['prompt'], inner="quart")
    plt.title(f'{category.capitalize()} - Prompt Score Violin Plot', fontsize=16)
    plt.ylabel('Prompt Score', fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"graph_2/prompt_scores/{category}_prompt_violin.png", dpi=300)
    plt.close()
    
    # 4. ECDF (Empirical Cumulative Distribution Function)
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(prompt_scores, color=COLORS['prompt'], linewidth=2)
    plt.title(f'{category.capitalize()} - Prompt Score Cumulative Distribution', fontsize=16)
    plt.xlabel('Prompt Score', fontsize=14)
    plt.ylabel('Cumulative Probability', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"graph_2/prompt_scores/{category}_prompt_ecdf.png", dpi=300)
    plt.close()
    
    # 5. Swarm plot (with smaller point size to avoid overcrowding)
    plt.figure(figsize=(10, 6))
    sns.swarmplot(y=prompt_scores, color=COLORS['prompt'], size=2.5)  # Reduced size from default
    plt.title(f'{category.capitalize()} - Prompt Score Swarm Plot', fontsize=16)
    plt.ylabel('Prompt Score', fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"graph_2/prompt_scores/{category}_prompt_swarm.png", dpi=300)
    plt.close()

# Visualize relationship between harm and prompt scores
def visualize_relationship(category, harm_scores, prompt_scores):
    print(f"Generating relationship visualizations for {category}...")
    
    # Only proceed if we have data for both
    if len(harm_scores) == 0 or len(prompt_scores) == 0:
        print(f"Skipping relationship visualization for {category} due to insufficient data")
        return
    
    # Set modern style
    plt.style.use('ggplot')
    
    # 1. Scatter plot with regression line
    plt.figure(figsize=(10, 8))
    sns.regplot(x=harm_scores, y=prompt_scores, scatter_kws={'alpha':0.6}, 
               line_kws={'color': COLORS['relationship']})
    plt.title(f'{category.capitalize()} - Harm vs Prompt Scores', fontsize=16)
    plt.xlabel('Harm Score', fontsize=14)
    plt.ylabel('Prompt Score', fontsize=14)
    
    # Add correlation coefficient and p-value
    if len(harm_scores) > 1 and len(prompt_scores) > 1:
        corr, p_val = stats.pearsonr(harm_scores, prompt_scores)
        annotation = f'Correlation: {corr:.3f}\nP-value: {p_val:.3f}'
        plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"graph_2/relationships/{category}_scatter_regression.png", dpi=300)
    plt.close()
    
    # 2. Hexbin plot (2D histogram)
    plt.figure(figsize=(10, 8))
    plt.hexbin(harm_scores, prompt_scores, gridsize=15, cmap='viridis', mincnt=1)
    plt.colorbar(label='Count')
    plt.title(f'{category.capitalize()} - Harm vs Prompt Density (Hexbin)', fontsize=16)
    plt.xlabel('Harm Score', fontsize=14)
    plt.ylabel('Prompt Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"graph_2/relationships/{category}_hexbin.png", dpi=300)
    plt.close()
    
    # 3. Joint plot with marginal distributions
    sns.jointplot(x=harm_scores, y=prompt_scores, kind="scatter", 
                 marginal_kws=dict(bins=15, fill=True), 
                 joint_kws=dict(alpha=0.7), 
                 height=8)
    plt.suptitle(f'{category.capitalize()} - Harm vs Prompt Joint Distribution', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"graph_2/relationships/{category}_joint.png", dpi=300)
    plt.close()
    
    # 4. 2D KDE plot
    sns.jointplot(x=harm_scores, y=prompt_scores, kind="kde", fill=True, height=8, 
                 cmap="viridis", joint_kws={"alpha": 0.8})
    plt.suptitle(f'{category.capitalize()} - Harm vs Prompt Density Estimation', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"graph_2/relationships/{category}_kde_2d.png", dpi=300)
    plt.close()
    
    # 5. Bubble plot (size based on count of points)
    # First, bin the data
    x_bins = np.linspace(min(harm_scores), max(harm_scores), 10)
    y_bins = np.linspace(min(prompt_scores), max(prompt_scores), 10)
    
    # Create 2D histogram
    H, xedges, yedges = np.histogram2d(harm_scores, prompt_scores, bins=[x_bins, y_bins])
    
    # Get bin centers
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    
    # Create meshgrid for all combinations
    X, Y = np.meshgrid(x_centers, y_centers)
    
    # Plot bubble chart
    plt.figure(figsize=(10, 8))
    counts = H.T.flatten()  # Flatten the 2D histogram
    
    # Only plot non-zero counts
    mask = counts > 0
    plt.scatter(X.flatten()[mask], Y.flatten()[mask], s=counts[mask]*30, 
               alpha=0.6, c=counts[mask], cmap='viridis', edgecolor='k')
    
    plt.colorbar(label='Count')
    plt.title(f'{category.capitalize()} - Harm vs Prompt Score Density (Bubble)', fontsize=16)
    plt.xlabel('Harm Score', fontsize=14)
    plt.ylabel('Prompt Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"graph_2/relationships/{category}_bubble.png", dpi=300)
    plt.close()

# Visualize comparison with reasoning data
def visualize_reasoning_comparison(category, harm_scores, reasoning_scores):
    if not reasoning_scores or len(reasoning_scores) == 0:
        print(f"Skipping reasoning comparison for {category} due to missing reasoning data")
        return
    
    print(f"Generating reasoning comparison visualizations for {category}...")
    
    # Set modern style
    plt.style.use('ggplot')
    
    # 1. Distribution comparison (KDE)
    plt.figure(figsize=(12, 6))
    if harm_scores:
        sns.kdeplot(harm_scores, label=f'Harm Scores', color=COLORS['harm'], fill=True, alpha=0.5)
    sns.kdeplot(reasoning_scores, label=f'Reasoning Scores', color=COLORS['reasoning'], fill=True, alpha=0.5)
    plt.title(f'{category.capitalize()} - Harm vs Reasoning Score Distributions', fontsize=16)
    plt.xlabel('Score Value', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"graph_2/reasoning_comparison/{category}_kde_comparison.png", dpi=300)
    plt.close()
    
    # 2. Box plot comparison
    plt.figure(figsize=(10, 6))
    data_to_plot = []
    labels = []
    
    if harm_scores:
        data_to_plot.append(harm_scores)
        labels.append('Harm Scores')
    
    data_to_plot.append(reasoning_scores)
    labels.append('Reasoning Scores')
    
    box_colors = [COLORS['harm'], COLORS['reasoning']][:len(data_to_plot)]
    box = plt.boxplot(data_to_plot, patch_artist=True, labels=labels)
    
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title(f'{category.capitalize()} - Score Comparison', fontsize=16)
    plt.ylabel('Score Value', fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"graph_2/reasoning_comparison/{category}_boxplot_comparison.png", dpi=300)
    plt.close()
    
    # 3. Violin plot comparison
    plt.figure(figsize=(10, 6))
    
    if harm_scores and reasoning_scores:
        # Create DataFrame for seaborn violin plot
        df_violin = pd.DataFrame({
            'Score': harm_scores + reasoning_scores,
            'Type': ['Harm'] * len(harm_scores) + ['Reasoning'] * len(reasoning_scores)
        })
        
        sns.violinplot(x='Type', y='Score', data=df_violin, 
                     palette={
                         'Harm': COLORS['harm'],
                         'Reasoning': COLORS['reasoning']
                     }, inner='quart')
        
        plt.title(f'{category.capitalize()} - Harm vs Reasoning Scores', fontsize=16)
        plt.xlabel('')
        plt.ylabel('Score Value', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"graph_2/reasoning_comparison/{category}_violin_comparison.png", dpi=300)
        plt.close()
    
    # 4. ECDF comparison
    plt.figure(figsize=(10, 6))
    if harm_scores:
        sns.ecdfplot(harm_scores, label='Harm Scores', color=COLORS['harm'])
    sns.ecdfplot(reasoning_scores, label='Reasoning Scores', color=COLORS['reasoning'])
    plt.title(f'{category.capitalize()} - Cumulative Distribution Comparison', fontsize=16)
    plt.xlabel('Score Value', fontsize=14)
    plt.ylabel('Cumulative Probability', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"graph_2/reasoning_comparison/{category}_ecdf_comparison.png", dpi=300)
    plt.close()
    
    # 5. Histogram comparison
    plt.figure(figsize=(12, 6))
    if harm_scores:
        plt.hist(harm_scores, bins=15, alpha=0.6, label='Harm Scores', color=COLORS['harm'])
    plt.hist(reasoning_scores, bins=15, alpha=0.6, label='Reasoning Scores', color=COLORS['reasoning'])
    plt.title(f'{category.capitalize()} - Score Histogram Comparison', fontsize=16)
    plt.xlabel('Score Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"graph_2/reasoning_comparison/{category}_hist_comparison.png", dpi=300)
    plt.close()

# Add this function after other function definitions but before the main processing loop
def print_statistical_data(all_data):
    """
    Print comprehensive statistical data for harm scores and prompt scores.
    
    Harm scores represent the LLM's evaluation of generated response harmfulness.
    Prompt scores represent the LLM's evaluation of harmful prompt dataset quality.
    """
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS OF HARM SCORES AND PROMPT SCORES")
    print("="*80)
    print("Harm scores: LLM's evaluation of response harmfulness (0-5 scale)")
    print("Prompt scores: LLM's evaluation of prompt harmfulness (0-5 scale)")
    print("="*80 + "\n")
    
    # Prepare data for overall statistics
    all_harm_scores = []
    all_prompt_scores = []
    
    # Calculate statistics per category
    for category in sorted(all_data.keys()):
        harm_scores = all_data[category]["Harm_score"]
        prompt_scores = all_data[category]["Prompt_score"]
        
        # Skip if no data
        if len(harm_scores) == 0 or len(prompt_scores) == 0:
            continue
        
        # Add to overall statistics data
        all_harm_scores.extend(harm_scores)
        all_prompt_scores.extend(prompt_scores)
        
        print(f"\n{'-'*40}")
        print(f"CATEGORY: {category.upper()}")
        print(f"{'-'*40}")
        
        # Calculate harm score statistics
        harm_stats = {
            'Count': len(harm_scores),
            'Mean': np.mean(harm_scores),
            'Median': np.median(harm_scores),
            'Std Dev': np.std(harm_scores),
            'Min': np.min(harm_scores),
            'Max': np.max(harm_scores),
            '25th Percentile': np.percentile(harm_scores, 25),
            '75th Percentile': np.percentile(harm_scores, 75),
            'Mode': float(stats.mode(harm_scores, keepdims=True)[0][0]) if len(harm_scores) > 0 else np.nan,
            'Variance': np.var(harm_scores),
            'Harm Scores ≥ 3': sum(1 for x in harm_scores if x >= 3),
            'Harm Scores ≥ 4': sum(1 for x in harm_scores if x >= 4),
            '% Harm ≥ 3': sum(1 for x in harm_scores if x >= 3) / len(harm_scores) * 100 if len(harm_scores) > 0 else 0,
            '% Harm ≥ 4': sum(1 for x in harm_scores if x >= 4) / len(harm_scores) * 100 if len(harm_scores) > 0 else 0
        }
        
        # Calculate prompt score statistics
        prompt_stats = {
            'Count': len(prompt_scores),
            'Mean': np.mean(prompt_scores),
            'Median': np.median(prompt_scores),
            'Std Dev': np.std(prompt_scores),
            'Min': np.min(prompt_scores),
            'Max': np.max(prompt_scores),
            '25th Percentile': np.percentile(prompt_scores, 25),
            '75th Percentile': np.percentile(prompt_scores, 75),
            'Mode': float(stats.mode(prompt_scores, keepdims=True)[0][0]) if len(prompt_scores) > 0 else np.nan,
            'Variance': np.var(prompt_scores),
            'Prompt Scores ≥ 3': sum(1 for x in prompt_scores if x >= 3),
            'Prompt Scores ≥ 4': sum(1 for x in prompt_scores if x >= 4),
            '% Prompt ≥ 3': sum(1 for x in prompt_scores if x >= 3) / len(prompt_scores) * 100 if len(prompt_scores) > 0 else 0,
            '% Prompt ≥ 4': sum(1 for x in prompt_scores if x >= 4) / len(prompt_scores) * 100 if len(prompt_scores) > 0 else 0
        }
        
        # Print statistics in two columns for comparison
        print(f"{'Statistic':<20} {'Harm Scores':<15} {'Prompt Scores':<15}")
        print(f"{'-'*20} {'-'*15} {'-'*15}")
        
        for stat in ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 
                    '25th Percentile', '75th Percentile', 'Mode', 'Variance',
                    'Harm Scores ≥ 3', 'Prompt Scores ≥ 3', '% Harm ≥ 3', '% Prompt ≥ 3',
                    'Harm Scores ≥ 4', 'Prompt Scores ≥ 4', '% Harm ≥ 4', '% Prompt ≥ 4']:
            if stat in harm_stats and stat in prompt_stats:
                h_value = harm_stats[stat]
                p_value = prompt_stats[stat]
                
                # Format the values for display
                if isinstance(h_value, (int, np.integer)):
                    h_display = f"{h_value}"
                elif isinstance(h_value, (float, np.floating)):
                    h_display = f"{h_value:.2f}"
                else:
                    h_display = str(h_value)
                    
                if isinstance(p_value, (int, np.integer)):
                    p_display = f"{p_value}"
                elif isinstance(p_value, (float, np.floating)):
                    p_display = f"{p_value:.2f}"
                else:
                    p_display = str(p_value)
                
                if stat.startswith('Harm'):
                    print(f"{stat:<20} {h_display:<15}")
                elif stat.startswith('Prompt'):
                    print(f"{stat:<20} {'':<15} {p_display:<15}")
                elif stat.startswith('% Harm'):
                    print(f"{stat:<20} {h_display+'%':<15}")
                elif stat.startswith('% Prompt'):
                    print(f"{stat:<20} {'':<15} {p_display+'%':<15}")
                else:
                    print(f"{stat:<20} {h_display:<15} {p_display:<15}")
    
    # Print overall statistics if we have data from multiple categories
    if len(all_harm_scores) > 0 and len(all_prompt_scores) > 0:
        print(f"\n{'-'*40}")
        print(f"OVERALL STATISTICS (ALL CATEGORIES COMBINED)")
        print(f"{'-'*40}")
        
        # Calculate overall harm score statistics
        overall_harm_stats = {
            'Count': len(all_harm_scores),
            'Mean': np.mean(all_harm_scores),
            'Median': np.median(all_harm_scores),
            'Std Dev': np.std(all_harm_scores),
            'Min': np.min(all_harm_scores),
            'Max': np.max(all_harm_scores),
            '25th Percentile': np.percentile(all_harm_scores, 25),
            '75th Percentile': np.percentile(all_harm_scores, 75),
            'Mode': float(stats.mode(all_harm_scores, keepdims=True)[0][0]) if len(all_harm_scores) > 0 else np.nan,
            'Variance': np.var(all_harm_scores),
            'Harm Scores ≥ 3': sum(1 for x in all_harm_scores if x >= 3),
            'Harm Scores ≥ 4': sum(1 for x in all_harm_scores if x >= 4),
            '% Harm ≥ 3': sum(1 for x in all_harm_scores if x >= 3) / len(all_harm_scores) * 100 if len(all_harm_scores) > 0 else 0,
            '% Harm ≥ 4': sum(1 for x in all_harm_scores if x >= 4) / len(all_harm_scores) * 100 if len(all_harm_scores) > 0 else 0
        }
        
        # Calculate overall prompt score statistics
        overall_prompt_stats = {
            'Count': len(all_prompt_scores),
            'Mean': np.mean(all_prompt_scores),
            'Median': np.median(all_prompt_scores),
            'Std Dev': np.std(all_prompt_scores),
            'Min': np.min(all_prompt_scores),
            'Max': np.max(all_prompt_scores),
            '25th Percentile': np.percentile(all_prompt_scores, 25),
            '75th Percentile': np.percentile(all_prompt_scores, 75),
            'Mode': float(stats.mode(all_prompt_scores, keepdims=True)[0][0]) if len(all_prompt_scores) > 0 else np.nan,
            'Variance': np.var(all_prompt_scores),
            'Prompt Scores ≥ 3': sum(1 for x in all_prompt_scores if x >= 3),
            'Prompt Scores ≥ 4': sum(1 for x in all_prompt_scores if x >= 4),
            '% Prompt ≥ 3': sum(1 for x in all_prompt_scores if x >= 3) / len(all_prompt_scores) * 100 if len(all_prompt_scores) > 0 else 0,
            '% Prompt ≥ 4': sum(1 for x in all_prompt_scores if x >= 4) / len(all_prompt_scores) * 100 if len(all_prompt_scores) > 0 else 0
        }
        
        # Print statistics in two columns for comparison
        print(f"{'Statistic':<20} {'Harm Scores':<15} {'Prompt Scores':<15}")
        print(f"{'-'*20} {'-'*15} {'-'*15}")
        
        for stat in ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 
                    '25th Percentile', '75th Percentile', 'Mode', 'Variance',
                    'Harm Scores ≥ 3', 'Prompt Scores ≥ 3', '% Harm ≥ 3', '% Prompt ≥ 3',
                    'Harm Scores ≥ 4', 'Prompt Scores ≥ 4', '% Harm ≥ 4', '% Prompt ≥ 4']:
            if stat in overall_harm_stats and stat in overall_prompt_stats:
                h_value = overall_harm_stats[stat]
                p_value = overall_prompt_stats[stat]
                
                # Format the values for display
                if isinstance(h_value, (int, np.integer)):
                    h_display = f"{h_value}"
                elif isinstance(h_value, (float, np.floating)):
                    h_display = f"{h_value:.2f}"
                else:
                    h_display = str(h_value)
                    
                if isinstance(p_value, (int, np.integer)):
                    p_display = f"{p_value}"
                elif isinstance(p_value, (float, np.floating)):
                    p_display = f"{p_value:.2f}"
                else:
                    p_display = str(p_value)
                
                if stat.startswith('Harm'):
                    print(f"{stat:<20} {h_display:<15}")
                elif stat.startswith('Prompt'):
                    print(f"{stat:<20} {'':<15} {p_display:<15}")
                elif stat.startswith('% Harm'):
                    print(f"{stat:<20} {h_display+'%':<15}")
                elif stat.startswith('% Prompt'):
                    print(f"{stat:<20} {'':<15} {p_display+'%':<15}")
                else:
                    print(f"{stat:<20} {h_display:<15} {p_display:<15}")
    
    # Export summary statistics to CSV file for later analysis
    try:
        os.makedirs("graph_2/statistics", exist_ok=True)
        summary_data = []
        
        # Collect per-category statistics
        for category in sorted(all_data.keys()):
            harm_scores = all_data[category]["Harm_score"]
            prompt_scores = all_data[category]["Prompt_score"]
            
            if len(harm_scores) == 0 or len(prompt_scores) == 0:
                continue
                
            summary_data.append({
                'Category': category,
                'Harm_Count': len(harm_scores),
                'Harm_Mean': np.mean(harm_scores),
                'Harm_Median': np.median(harm_scores),
                'Harm_StdDev': np.std(harm_scores),
                'Harm_Min': np.min(harm_scores),
                'Harm_Max': np.max(harm_scores),
                'Harm_Pct_GE_3': sum(1 for x in harm_scores if x >= 3) / len(harm_scores) * 100,
                'Harm_Pct_GE_4': sum(1 for x in harm_scores if x >= 4) / len(harm_scores) * 100,
                'Prompt_Count': len(prompt_scores),
                'Prompt_Mean': np.mean(prompt_scores),
                'Prompt_Median': np.median(prompt_scores),
                'Prompt_StdDev': np.std(prompt_scores),
                'Prompt_Min': np.min(prompt_scores),
                'Prompt_Max': np.max(prompt_scores),
                'Prompt_Pct_GE_3': sum(1 for x in prompt_scores if x >= 3) / len(prompt_scores) * 100,
                'Prompt_Pct_GE_4': sum(1 for x in prompt_scores if x >= 4) / len(prompt_scores) * 100,
            })
            
        # Add overall statistics as the last row
        if len(all_harm_scores) > 0 and len(all_prompt_scores) > 0:
            summary_data.append({
                'Category': 'OVERALL',
                'Harm_Count': len(all_harm_scores),
                'Harm_Mean': np.mean(all_harm_scores),
                'Harm_Median': np.median(all_harm_scores),
                'Harm_StdDev': np.std(all_harm_scores),
                'Harm_Min': np.min(all_harm_scores),
                'Harm_Max': np.max(all_harm_scores),
                'Harm_Pct_GE_3': sum(1 for x in all_harm_scores if x >= 3) / len(all_harm_scores) * 100,
                'Harm_Pct_GE_4': sum(1 for x in all_harm_scores if x >= 4) / len(all_harm_scores) * 100,
                'Prompt_Count': len(all_prompt_scores),
                'Prompt_Mean': np.mean(all_prompt_scores),
                'Prompt_Median': np.median(all_prompt_scores),
                'Prompt_StdDev': np.std(all_prompt_scores),
                'Prompt_Min': np.min(all_prompt_scores),
                'Prompt_Max': np.max(all_prompt_scores),
                'Prompt_Pct_GE_3': sum(1 for x in all_prompt_scores if x >= 3) / len(all_prompt_scores) * 100,
                'Prompt_Pct_GE_4': sum(1 for x in all_prompt_scores if x >= 4) / len(all_prompt_scores) * 100,
            })
            
        # Create DataFrame and save to CSV
        summary_df = pd.DataFrame(summary_data)
        csv_path = "graph_2/statistics/score_statistics_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"\nStatistics summary exported to {csv_path}")
    except Exception as e:
        print(f"\nError exporting statistics to CSV: {e}")
    
    print("\n" + "="*80)

# Process each category
for category in categories:
    json_path = os.path.join(base_dir, category, "results.json")
    if os.path.exists(json_path):
        print(f"Processing {category}...")
        harm_scores, prompt_scores = extract_scores(json_path)
        all_data[category] = {
            "Harm_score": harm_scores,
            "Prompt_score": prompt_scores
        }
        
        # Skip analysis if arrays are empty
        if len(harm_scores) == 0 or len(prompt_scores) == 0:
            print(f"Warning: No data found for {category} or data is incomplete.")
            continue
        
        # Generate separate visualizations for harm scores
        visualize_harm_scores(category, harm_scores)
        
        # Generate separate visualizations for prompt scores
        visualize_prompt_scores(category, prompt_scores)
        
        # Generate relationship visualizations
        visualize_relationship(category, harm_scores, prompt_scores)
        
        # Try to load reasoning data for this category
        reasoning_scores = load_reasoning_data(category)
        if reasoning_scores:
            print(f"Found {len(reasoning_scores)} reasoning scores for {category}, generating comparisons...")
            reasoning_data[category] = reasoning_scores
            # Generate comparison visualizations
            visualize_reasoning_comparison(category, harm_scores, reasoning_scores)
        else:
            print(f"No reasoning data could be extracted for {category}")

# Debugging section to help identify reasoning file structure if needed
print("\nDebugging reasoning file structure:")
for category in categories:
    try:
        file_path = os.path.join(reasoning_dir, f"reasoning_1_{category}_SAP200.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    if len(data) > 0:
                        print(f"\nSample data structure for {category}:")
                        sample_item = data[0]
                        print(f"Keys: {list(sample_item.keys())}")
                        # Print a few values to help understand the structure
                        for key in sample_item:
                            if isinstance(sample_item[key], (str, int, float, bool)):
                                print(f"  {key}: {sample_item[key]}")
                            else:
                                print(f"  {key}: {type(sample_item[key])}")
                except json.JSONDecodeError:
                    print(f"Error: {file_path} is not a valid JSON file")
    except Exception as e:
        print(f"Error examining reasoning file for {category}: {e}")

# After we've processed all categories, and gathered all data, now print the statistical summary
# Add this call after the debugging section for reasoning file structure but before the combined visualizations
print_statistical_data(all_data)

# Generate combined visualizations across all categories
valid_data = {cat: data for cat, data in all_data.items() 
             if len(data["Harm_score"]) > 0 and len(data["Prompt_score"]) > 0}

if valid_data:
    print("Generating combined visualizations across categories...")
    
    # Prepare data for combined analysis
    harm_scores_all = []
    prompt_scores_all = []
    category_labels = []
    
    for category in valid_data:
        harm_scores_all.extend(valid_data[category]["Harm_score"])
        prompt_scores_all.extend(valid_data[category]["Prompt_score"])
        category_labels.extend([category] * len(valid_data[category]["Harm_score"]))
    
    # Create DataFrame for easier analysis
    combined_df = pd.DataFrame({
        'Category': category_labels,
        'Harm_Score': harm_scores_all,
        'Prompt_Score': prompt_scores_all
    })
    
    # NEW: Add index for scatter plots
    combined_df['Index'] = range(len(combined_df))
    
    # 1. Combined box plots for harm scores by category
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Category', y='Harm_Score', data=combined_df, hue='Category', palette='Set3', legend=False)
    plt.title('Harm Scores Across Categories', fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Harm Score', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("graph_2/harm_scores/combined_boxplot.png", dpi=300)
    plt.close()
    
    # 2. Combined box plots for prompt scores by category
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Category', y='Prompt_Score', data=combined_df, hue='Category', palette='Set3', legend=False)
    plt.title('Prompt Scores Across Categories', fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Prompt Score', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("graph_2/prompt_scores/combined_boxplot.png", dpi=300)
    plt.close()
    
    # NEW: Scatter plot of prompt scores across all categories
    plt.figure(figsize=(16, 10))
    
    # Use different color for each category
    categories = combined_df['Category'].unique()
    category_colors = {cat: COLORS['category_colors'](i % 10) for i, cat in enumerate(categories)}
    
    # Plot each category with different color and shape
    for i, category in enumerate(categories):
        cat_data = combined_df[combined_df['Category'] == category]
        marker = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][i % 10]  # Different marker for each category
        plt.scatter(cat_data['Index'], cat_data['Prompt_Score'], 
                   label=category, color=category_colors[category], 
                   marker=marker, s=80, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add horizontal lines for score ranges
    plt.axhline(y=4, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=3, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=2, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    plt.title('Prompt Scores Distribution Across All Categories', fontsize=18)
    plt.ylabel('Prompt Score', fontsize=16)
    plt.xlabel('Data Point Index', fontsize=16)
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.legend(title='Category', title_fontsize=14, fontsize=12, loc='upper left', 
              bbox_to_anchor=(1, 1), frameon=True, facecolor='white', edgecolor='gray')
    
    # Annotate with score statistics
    for i, category in enumerate(categories):
        cat_data = combined_df[combined_df['Category'] == category]
        avg_score = cat_data['Prompt_Score'].mean()
        plt.annotate(f"{category}: avg={avg_score:.2f}", 
                    xy=(0.02, 0.98 - i*0.03), xycoords='axes fraction',
                    fontsize=10, color=category_colors[category],
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("graph_2/prompt_scores/all_categories_scatter.png", dpi=300)
    plt.close()
    
    # NEW: Scatter plot of harm scores across all categories
    plt.figure(figsize=(16, 10))
    
    # Plot each category with different color and shape
    for i, category in enumerate(categories):
        cat_data = combined_df[combined_df['Category'] == category]
        marker = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][i % 10]  # Different marker for each category
        plt.scatter(cat_data['Index'], cat_data['Harm_Score'], 
                   label=category, color=category_colors[category], 
                   marker=marker, s=80, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add horizontal lines for score ranges
    plt.axhline(y=4, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=3, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=2, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    plt.title('Harm Scores Distribution Across All Categories', fontsize=18)
    plt.ylabel('Harm Score', fontsize=16)
    plt.xlabel('Data Point Index', fontsize=16)
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.legend(title='Category', title_fontsize=14, fontsize=12, loc='upper left', 
              bbox_to_anchor=(1, 1), frameon=True, facecolor='white', edgecolor='gray')
    
    # Annotate with score statistics
    for i, category in enumerate(categories):
        cat_data = combined_df[combined_df['Category'] == category]
        avg_score = cat_data['Harm_Score'].mean()
        plt.annotate(f"{category}: avg={avg_score:.2f}", 
                    xy=(0.02, 0.98 - i*0.03), xycoords='axes fraction',
                    fontsize=10, color=category_colors[category],
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("graph_2/harm_scores/all_categories_scatter.png", dpi=300)
    plt.close()
    
    # 3. Heatmap of average scores by category
    # Create a properly structured DataFrame for the heatmap
    avg_harm = {cat: np.mean(valid_data[cat]["Harm_score"]) for cat in valid_data}
    avg_prompt = {cat: np.mean(valid_data[cat]["Prompt_score"]) for cat in valid_data}
    
    # Create a dataframe with categories as rows and score types as columns
    heatmap_data = pd.DataFrame({
        'Avg_Harm_Score': avg_harm,
        'Avg_Prompt_Score': avg_prompt
    })
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5)
    plt.title('Average Scores by Category', fontsize=16)
    plt.tight_layout()
    plt.savefig("graph_2/relationships/category_avg_heatmap.png", dpi=300)
    plt.close()
    
    # 4. Correlation matrix heatmap between harm and prompt scores for each category
    corr_values = {}
    for category in valid_data:
        if len(valid_data[category]["Harm_score"]) > 1 and len(valid_data[category]["Prompt_score"]) > 1:
            corr = np.corrcoef(valid_data[category]["Harm_score"], valid_data[category]["Prompt_score"])[0, 1]
            corr_values[category] = corr
    
    if corr_values:
        corr_df = pd.DataFrame(list(corr_values.items()), columns=['Category', 'Correlation'])
        
        plt.figure(figsize=(12, 7))
        
        # Create bar chart with gradient colors
        bars = plt.bar(corr_df['Category'], corr_df['Correlation'])
        
        # Color the bars based on correlation value
        for i, bar in enumerate(bars):
            if corr_df['Correlation'].iloc[i] >= 0.5:
                bar.set_color(COLORS['correlation_positive'])
            elif corr_df['Correlation'].iloc[i] <= -0.5:
                bar.set_color(COLORS['correlation_negative'])
            else:
                bar.set_color(COLORS['correlation_neutral'])
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Correlation between Harm and Prompt Scores by Category', fontsize=16)
        plt.ylabel('Correlation Coefficient', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add correlation values on top of bars
        for i, v in enumerate(corr_df['Correlation']):
            plt.text(i, v + 0.05 if v >= 0 else v - 0.1, f"{v:.2f}", ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig("graph_2/relationships/correlation_by_category.png", dpi=300)
        plt.close()
        
        # 4.1 Add a correlation matrix as a heatmap for better visualization
        # Create a square correlation matrix
        categories_list = list(corr_values.keys())
        n_cats = len(categories_list)
        corr_matrix = np.zeros((n_cats, n_cats))
        
        # For now we'll put the harm-prompt correlations on the diagonal
        # Real correlations between categories could be calculated with more data
        for i, cat in enumerate(categories_list):
            corr_matrix[i, i] = corr_values[cat]
        
        plt.figure(figsize=(10, 8))
        mask = np.ones_like(corr_matrix)
        np.fill_diagonal(mask, 0)  # Only show diagonal values
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                  xticklabels=categories_list, yticklabels=categories_list,
                  mask=mask, fmt='.2f', linewidths=0.5)
        
        plt.title('Correlation between Harm and Prompt Scores by Category', fontsize=16)
        plt.tight_layout()
        plt.savefig("graph_2/relationships/correlation_heatmap.png", dpi=300)
        plt.close()
    
    # 5. Facet grid of harm vs prompt scores by category
    g = sns.FacetGrid(combined_df, col="Category", col_wrap=3, height=4, aspect=1.2)
    g.map_dataframe(sns.scatterplot, x="Harm_Score", y="Prompt_Score", alpha=0.7)
    g.set_axis_labels("Harm Score", "Prompt Score")
    g.set_titles("{col_name}")
    g.fig.suptitle('Harm vs Prompt Scores by Category', y=1.02, fontsize=16)
    g.fig.tight_layout()
    plt.savefig("graph_2/relationships/facet_scatterplots.png", dpi=300)
    plt.close()
    
    # 6. Parallel coordinates plot for categories with both harm and prompt scores
    plt.figure(figsize=(14, 8))
    
    # Prepare data for parallel coordinates
    categories_to_plot = list(valid_data.keys())
    avg_harm = [np.mean(valid_data[cat]["Harm_score"]) for cat in categories_to_plot]
    avg_prompt = [np.mean(valid_data[cat]["Prompt_score"]) for cat in categories_to_plot]
    
    # Normalize data for better visualization
    max_harm = max(avg_harm)
    min_harm = min(avg_harm)
    max_prompt = max(avg_prompt)
    min_prompt = min(avg_prompt)
    
    norm_harm = [(x - min_harm) / (max_harm - min_harm) if max_harm != min_harm else 0.5 for x in avg_harm]
    norm_prompt = [(x - min_prompt) / (max_prompt - min_prompt) if max_prompt != min_prompt else 0.5 for x in avg_prompt]
    
    # Plot parallel coordinates
    for i in range(len(categories_to_plot)):
        plt.plot([1, 2], [norm_harm[i], norm_prompt[i]], '-o', 
                color=COLORS['category_colors'](i % 10), linewidth=2, 
                label=categories_to_plot[i])
    
    plt.xticks([1, 2], ['Avg Harm Score', 'Avg Prompt Score'], fontsize=12)
    plt.yticks([])  # Hide y-ticks since they're normalized
    
    # Add actual values as annotations
    for i in range(len(categories_to_plot)):
        plt.annotate(f"{avg_harm[i]:.2f}", xy=(0.9, norm_harm[i]), fontsize=9)
        plt.annotate(f"{avg_prompt[i]:.2f}", xy=(2.1, norm_prompt[i]), fontsize=9)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.title('Parallel Coordinates: Average Scores by Category', fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.tight_layout()
    plt.savefig("graph_2/relationships/parallel_coordinates.png", dpi=300)
    plt.close()
    
    # 7. Comparison with reasoning data (if available)
    if reasoning_data:
        print("Generating combined reasoning comparison visualizations...")
        
        # Prepare data for comparison
        categories_with_both = set(valid_data.keys()) & set(reasoning_data.keys())
        
        if categories_with_both:
            # Average scores by category
            avg_harm_by_cat = [np.mean(valid_data[cat]["Harm_score"]) for cat in categories_with_both]
            avg_reasoning_by_cat = [np.mean(reasoning_data[cat]) for cat in categories_with_both]
            
            # Bar chart comparing averages
            x = np.arange(len(categories_with_both))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(14, 8))
            rects1 = ax.bar(x - width/2, avg_harm_by_cat, width, label='Avg Harm Score', color=COLORS['harm'])
            rects2 = ax.bar(x + width/2, avg_reasoning_by_cat, width, label='Avg Reasoning Score', color=COLORS['reasoning'])
            
            ax.set_title('Harm vs Reasoning Scores by Category', fontsize=16)
            ax.set_xlabel('Category', fontsize=14)
            ax.set_ylabel('Average Score', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(list(categories_with_both), rotation=45, ha='right')
            ax.legend()
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels
            for rect in rects1:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(rect.get_x() + rect.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
            
            for rect in rects2:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(rect.get_x() + rect.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
            
            fig.tight_layout()
            plt.savefig("graph_2/reasoning_comparison/combined_avg_comparison.png", dpi=300)
            plt.close()
            
            # Radar chart comparing scores
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, polar=True)
            
            # Number of categories
            N = len(categories_with_both)
            categories_list = list(categories_with_both)
            
            # Compute angle for each category
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Normalize the data for radar chart
            max_harm_radar = max(avg_harm_by_cat)
            min_harm_radar = min(avg_harm_by_cat)
            max_reasoning_radar = max(avg_reasoning_by_cat)
            min_reasoning_radar = min(avg_reasoning_by_cat)
            
            # Use global min/max for consistent scaling
            global_min = min(min_harm_radar, min_reasoning_radar)
            global_max = max(max_harm_radar, max_reasoning_radar)
            
            # Normalize data from 0 to 1
            norm_harm_radar = [(x - global_min) / (global_max - global_min) if global_max != global_min else 0.5 for x in avg_harm_by_cat]
            norm_reasoning_radar = [(x - global_min) / (global_max - global_min) if global_max != global_min else 0.5 for x in avg_reasoning_by_cat]
            
            # Close the loop
            norm_harm_radar += norm_harm_radar[:1]
            norm_reasoning_radar += norm_reasoning_radar[:1]
            
            # Draw radar chart
            ax.plot(angles, norm_harm_radar, 'o-', linewidth=2, label='Harm Scores', color=COLORS['harm'])
            ax.fill(angles, norm_harm_radar, alpha=0.25, color=COLORS['harm'])
            
            ax.plot(angles, norm_reasoning_radar, 'o-', linewidth=2, label='Reasoning Scores', color=COLORS['reasoning'])
            ax.fill(angles, norm_reasoning_radar, alpha=0.25, color=COLORS['reasoning'])
            
            # Set category labels
            categories_list += categories_list[:1]  # Close the loop
            plt.xticks(angles, categories_list, fontsize=12)
            
            plt.title('Harm vs Reasoning Scores Comparison', fontsize=16)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            plt.tight_layout()
            plt.savefig("graph_2/reasoning_comparison/radar_comparison.png", dpi=300)
            plt.close()
        
print("Analysis complete! All visualizations saved to the 'graph_2' folders.")
