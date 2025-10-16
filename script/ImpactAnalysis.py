"""
Economic Impact Report Generator
Generates LaTeX report with histograms and tables for budget allocation models
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Configuration
MODELS = [1, 2, 3, 4, 5, 6, 9]
BASE_DIR = Path("../report")
MODEL_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"
BASE_YEAR=2025

# Create directories if they don't exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_model_data(model_num, base_year=BASE_YEAR):
    """Load full-population predictions and metrics for a specific model."""
    model_path = MODEL_DIR / f"model_{model_num}"
    
    # Look for full-population file first
    pred_file = model_path / f"predictions_full_{base_year}.csv"
    if not pred_file.exists():
        # Fallback to the old train/test predictions.csv if needed
        pred_file = model_path / "predictions.csv"
        print(f"[Warning] Using {pred_file.name} (full-population file not found for Model {model_num})")
    else:
        print(f"[Info] Loaded full-population predictions for Model {model_num} ({pred_file.name})")
    
    # Load predictions CSV
    predictions = pd.read_csv(pred_file)
    
    # Standardize column names
    predictions.columns = [c.strip().lower() for c in predictions.columns]
    rename_map = {
        'actual': 'actual',
        'predicted': 'predicted',
        'error': 'error'
    }
    # If error column not present, compute it
    if 'error' not in predictions.columns and {'actual', 'predicted'}.issubset(predictions.columns):
        predictions['error'] = predictions['predicted'] - predictions['actual']
    
    # Load metrics JSON (for R², RMSE, etc.)
    metrics_file = model_path / "metrics.json"
    metrics = {}
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    else:
        print(f"[Warning] metrics.json not found for Model {model_num}")
    
    return predictions, metrics



def calculate_conservative_estimates(predictions_df):
    """Calculate conservative budget estimates (max of actual or predicted)."""
    conservative = np.maximum(
        predictions_df['actual'].values,
        predictions_df['predicted'].values
    )
    return conservative


def calculate_impact_metrics(predictions_df, conservative):
    """Calculate economic impact statistics."""
    actual = predictions_df['actual'].values
    predicted = predictions_df['predicted'].values
    error = predictions_df['error'].values
    
    metrics = {
        'n_samples': len(predictions_df),
        'total_actual': np.sum(actual),
        'total_predicted': np.sum(predicted),
        'total_conservative': np.sum(conservative),
        'mean_actual': np.mean(actual),
        'mean_predicted': np.mean(predicted),
        'mean_conservative': np.mean(conservative),
        'median_actual': np.median(actual),
        'median_predicted': np.median(predicted),
        'median_conservative': np.median(conservative),
        'std_actual': np.std(actual),
        'std_predicted': np.std(predicted),
        'std_conservative': np.std(conservative),
        'economic_impact': np.sum(conservative) - np.sum(actual),
        'impact_percentage': 100 * (np.sum(conservative) - np.sum(actual)) / np.sum(actual),
        'over_budget_cases': np.sum(predicted > actual),
        'over_budget_pct': 100 * np.sum(predicted > actual) / len(actual),
    }
    
    return metrics


def create_age_groups(age):
    """Categorize age into groups."""
    if age < 21:
        return 'Under 21'
    elif age <= 30:
        return '21-30'
    else:
        return '31+'


def create_impact_level(conservative_val, actual_val):
    """Categorize the level of budget increase/decrease."""
    impact = conservative_val - actual_val
    impact_pct = 100 * impact / actual_val if actual_val > 0 else 0
    
    if impact_pct < -10:
        return 'Large Decrease (>10\%)'
    elif impact_pct < 0:
        return 'Small Decrease (0-10\%)'
    elif impact_pct == 0:
        return 'No Change'
    elif impact_pct <= 10:
        return 'Small Increase (0-10\%)'
    elif impact_pct <= 25:
        return 'Moderate Increase (10-25\%)'
    else:
        return 'Large Increase (>25\%)'


def analyze_subgroups(predictions_df, conservative):
    """Analyze economic impact by different subgroups."""
    df = predictions_df.copy()
    df['conservative'] = conservative
    df['impact'] = conservative - df['actual'].values
    df['impact_pct'] = 100 * df['impact'] / df['actual']
    
    # Create grouping variables
    if 'age' in df.columns:
        df['age_group'] = df['age'].apply(create_age_groups)
    
    # Cost quartiles based on actual amounts
    df['cost_quartile'] = pd.qcut(df['actual'], q=4, 
                                   labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'],
                                   duplicates='drop')
    
    # Impact levels
    df['impact_level'] = df.apply(
        lambda row: create_impact_level(row['conservative'], row['actual']), 
        axis=1
    )
    
    subgroup_results = {}
    
    # Age group analysis
    if 'age_group' in df.columns:
        age_analysis = df.groupby('age_group').agg({
            'actual': ['count', 'sum', 'mean'],
            'conservative': ['sum', 'mean'],
            'impact': ['sum', 'mean'],
            'impact_pct': 'mean'
        }).round(2)
        subgroup_results['age'] = age_analysis
    
    # Living setting analysis
    if 'living_setting' in df.columns:
        living_analysis = df.groupby('living_setting').agg({
            'actual': ['count', 'sum', 'mean'],
            'conservative': ['sum', 'mean'],
            'impact': ['sum', 'mean'],
            'impact_pct': 'mean'
        }).round(2)
        subgroup_results['living'] = living_analysis
    
    # Cost quartile analysis
    quartile_analysis = df.groupby('cost_quartile').agg({
        'actual': ['count', 'sum', 'mean'],
        'conservative': ['sum', 'mean'],
        'impact': ['sum', 'mean'],
        'impact_pct': 'mean'
    }).round(2)
    subgroup_results['cost_quartile'] = quartile_analysis
    
    # Impact level analysis
    impact_analysis = df.groupby('impact_level').agg({
        'actual': ['count', 'sum', 'mean'],
        'conservative': ['sum', 'mean'],
        'impact': ['sum', 'mean'],
        'impact_pct': 'mean'
    }).round(2)
    # Reorder impact levels logically
    impact_order = ['Large Decrease (>10\%)', 'Small Decrease (0-10\%)', 'No Change',
                    'Small Increase (0-10\%)', 'Moderate Increase (10-25\%)', 
                    'Large Increase (>25\%)']
    impact_analysis = impact_analysis.reindex(
        [idx for idx in impact_order if idx in impact_analysis.index]
    )
    subgroup_results['impact_level'] = impact_analysis
    
    return subgroup_results


def create_histograms(predictions_df, conservative, model_num):
    """Create 4-subplot histogram figure for a model."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Model {model_num}: Economic Impact Analysis', 
                 fontsize=14, fontweight='bold')
    
    actual = predictions_df['actual'].values
    predicted = predictions_df['predicted'].values
    error = predictions_df['error'].values
    
    # Plot 1: Actual costs
    axes[0, 0].hist(actual / 1000, bins=50, color='#005F9E', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Actual Cost ($1000s)', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].set_title('Distribution of Actual Costs', fontsize=11, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].axvline(np.mean(actual) / 1000, color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: ${np.mean(actual)/1000:.2f}K')
    axes[0, 0].legend()
    
    # Plot 2: Predicted costs
    axes[0, 1].hist(predicted / 1000, bins=50, color='#D35400', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Predicted Cost ($1000s)', fontsize=10)
    axes[0, 1].set_ylabel('Frequency', fontsize=10)
    axes[0, 1].set_title('Distribution of Predicted Costs', fontsize=11, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].axvline(np.mean(predicted) / 1000, color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: ${np.mean(predicted)/1000:.2f}K')
    axes[0, 1].legend()
    
    # Plot 3: Prediction errors
    axes[1, 0].hist(error / 1000, bins=50, color='#5E35B1', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Prediction Error ($1000s)', fontsize=10)
    axes[1, 0].set_ylabel('Frequency', fontsize=10)
    axes[1, 0].set_title('Distribution of Prediction Errors', fontsize=11, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1, 0].axvline(np.mean(error) / 1000, color='orange', linestyle='--', 
                       linewidth=2, label=f'Mean: ${np.mean(error)/1000:.2f}K')
    axes[1, 0].legend()
    
    # Plot 4: Conservative estimates
    axes[1, 1].hist(conservative / 1000, bins=50, color='#00695C', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Conservative Budget ($1000s)', fontsize=10)
    axes[1, 1].set_ylabel('Frequency', fontsize=10)
    axes[1, 1].set_title('Distribution of Conservative Budget Estimates', 
                         fontsize=11, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].axvline(np.mean(conservative) / 1000, color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: ${np.mean(conservative)/1000:.2f}K')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save figure
    output_file = FIGURES_DIR / f"model_{model_num}_Impact_Histograms.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file


def format_number(num, decimals=2):
    """Format number for LaTeX with thousand separators."""
    if abs(num) >= 1000:
        return f"{num:,.{decimals}f}"
    else:
        return f"{num:.{decimals}f}"


def generate_latex_output(all_results):
    """Generate comprehensive LaTeX report."""
    output_file = LOGS_DIR / "ImpactAnalysisCalculations.tex"
    
    with open(output_file, 'w') as f:
        # Header
        f.write(f"\\renewcommand{{\\FiscalYear}}{{(fiscal year {BASE_YEAR-1}--{BASE_YEAR})}}\n")
        f.write("% Economic Impact Analysis Report\n")
        f.write(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("% Conservative Budget Estimation Analysis\n\n")
        
        f.write("\\section{Economic Impact Analysis}\n\\label{sec:economic_impact}\n\n")
        
        f.write("This section presents the economic impact analysis for each budget ")
        f.write("allocation model. The conservative budget estimate is defined as the ")
        f.write("maximum of the actual cost and predicted cost for each case: ")
        f.write("$\\text{Conservative} = \\max(\\text{Actual}, \\text{Predicted})$. ")
        f.write("This approach ensures adequate funding while accounting for model uncertainty.\n\n")
        
        # Individual model sections
        for model_num in MODELS:
            if model_num not in all_results:
                continue
            
            result = all_results[model_num]
            metrics = result['impact_metrics']
            model_metrics = result['model_metrics']
            
            f.write(f"\\subsection{{Model {model_num}: Impact Analysis}}\n")
            f.write(f"\\label{{subsec:model{model_num}_impact}}\n\n")
            
            # Summary table
            f.write(f"\\begin{{table}}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\small\n")
            f.write(f"\\caption{{Model {model_num}: Economic Impact Summary \\FiscalYear}}\n")
            f.write(f"\\label{{tab:model{model_num}_impact_summary}}\n")
            f.write("\\begin{tabular}{lrr}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Metric} & \\textbf{Value} & \\textbf{Per Client} \\\\\n")
            f.write("\\midrule\n")
            
            # Sample size
            f.write(f"Sample Size & {metrics['n_samples']:,} & --- \\\\\n")
            f.write("\\midrule\n")
            
            # Total costs
            f.write(f"Total Actual Cost & \\${format_number(metrics['total_actual'])} & ")
            f.write(f"\\${format_number(metrics['mean_actual'])} \\\\\n")
            
            f.write(f"Total Predicted Cost & \\${format_number(metrics['total_predicted'])} & ")
            f.write(f"\\${format_number(metrics['mean_predicted'])} \\\\\n")
            
            f.write(f"Total Conservative Budget & \\${format_number(metrics['total_conservative'])} & ")
            f.write(f"\\${format_number(metrics['mean_conservative'])} \\\\\n")
            
            f.write("\\midrule\n")
            
            # Economic impact
            impact_sign = "+" if metrics['economic_impact'] >= 0 else ""
            f.write(f"\\textbf{{Economic Impact}} & ")
            f.write(f"\\textbf{{\\${impact_sign}{format_number(metrics['economic_impact'])}}} & ")
            f.write(f"\\textbf{{\\${impact_sign}{format_number(metrics['economic_impact']/metrics['n_samples'])}}} \\\\\n")
            
            f.write(f"Impact Percentage & {format_number(metrics['impact_percentage'], 2)}\\% & --- \\\\\n")
            
            f.write("\\midrule\n")
            
            # Additional statistics
            f.write(f"Cases Over Budget & {metrics['over_budget_cases']:,} & ")
            f.write(f"{format_number(metrics['over_budget_pct'], 1)}\\% \\\\\n")
            
            f.write("\\midrule\n")
            
            # Model performance from metrics.json
            f.write(f"Model $R^2$ (Test) & {model_metrics.get('r2_test', 0):.4f} & --- \\\\\n")
            f.write(f"RMSE (Test) & \\${format_number(model_metrics.get('rmse_test', 0))} & --- \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            
            # Subgroup Analysis Tables
            subgroup_results = result['subgroup_results']
            
            # Age Group Analysis
            if 'age' in subgroup_results:
                f.write(f"\\begin{{table}}[htbp]\n")
                f.write("\\centering\n")
                f.write("\\small\n")
                f.write(f"\\caption{{Model {model_num}: Economic Impact by Age Group \\FiscalYear}}\n")
                f.write(f"\\label{{tab:model{model_num}_impact_summary}}\n")
                f.write("\\begin{tabular}{lrrrrrr}\n")
                f.write("\\toprule\n")
                f.write("\\textbf{Age Group} & \\textbf{N} & \\textbf{\\%} & \\textbf{Mean Actual} & ")
                f.write("\\textbf{Mean Conservative} & \\textbf{Impact} & \\textbf{Impact \\%} \\\\\n")
                f.write("\\midrule\n")
                
                age_df = subgroup_results['age']
                total_n = age_df[('actual', 'count')].sum()
                
                for idx in age_df.index:
                    n = int(age_df.loc[idx, ('actual', 'count')])
                    pct = 100 * n / total_n
                    mean_actual = age_df.loc[idx, ('actual', 'mean')]
                    mean_cons = age_df.loc[idx, ('conservative', 'mean')]
                    impact = age_df.loc[idx, ('impact', 'sum')]
                    impact_pct = age_df.loc[idx, ('impact_pct', 'mean')]
                    
                    f.write(f"{idx} & {n:,} & {format_number(pct, 1)}\\% & ")
                    f.write(f"\\${format_number(mean_actual)} & ")
                    f.write(f"\\${format_number(mean_cons)} & ")
                    impact_sign = "+" if impact >= 0 else ""
                    f.write(f"\\${impact_sign}{format_number(impact)} & ")
                    f.write(f"{impact_sign}{format_number(impact_pct, 2)}\\% \\\\\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n\n")
            
            # Living Setting Analysis
            if 'living' in subgroup_results:
                f.write(f"\\begin{{table}}[htbp]\n")
                f.write("\\centering\n")
                f.write("\\small\n")
                f.write(f"\\caption{{Model {model_num}: Economic Impact by Living Setting \\FiscalYear}}\n")
                f.write(f"\\label{{tab:model{model_num}_impact_living}}\n")
                f.write("\\begin{tabular}{lrrrrrr}\n")
                f.write("\\toprule\n")
                f.write("\\textbf{\\shortstack{Living \\\\ Setting}} & \\textbf{N} & \\textbf{\\%} & \\textbf{Mean Actual} & ")
                f.write("\\textbf{Mean Conservative} & \\textbf{Impact} & \\textbf{Impact \\%} \\\\\n")
    
                f.write("\\midrule\n")
                
                living_df = subgroup_results['living']
                total_n = living_df[('actual', 'count')].sum()
                
                for idx in living_df.index:
                    n = int(living_df.loc[idx, ('actual', 'count')])
                    pct = 100 * n / total_n
                    mean_actual = living_df.loc[idx, ('actual', 'mean')]
                    mean_cons = living_df.loc[idx, ('conservative', 'mean')]
                    impact = living_df.loc[idx, ('impact', 'sum')]
                    impact_pct = living_df.loc[idx, ('impact_pct', 'mean')]
                    
                    f.write(f"{idx} & {n:,} & {format_number(pct, 1)}\\% & ")
                    f.write(f"\\${format_number(mean_actual)} & ")
                    f.write(f"\\${format_number(mean_cons)} & ")
                    impact_sign = "+" if impact >= 0 else ""
                    f.write(f"\\${impact_sign}{format_number(impact)} & ")
                    f.write(f"{impact_sign}{format_number(impact_pct, 2)}\\% \\\\\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n\n")
            
            # Cost Quartile Analysis
            if 'cost_quartile' in subgroup_results:
                f.write(f"\\begin{{table}}[htbp]\n")
                f.write("\\centering\n")
                f.write("\\small\n")
                f.write(f"\\caption{{Model {model_num}: Economic Impact by Budget Quartile \\FiscalYear}}\n")
                f.write(f"\\label{{tab:model{model_num}_impact_quartile}}\n")
                f.write("\\begin{tabular}{lrrrrr}\n")
                f.write("\\toprule\n")
                f.write("\\textbf{Budget Quartile} & \\textbf{N} & \\textbf{Mean Actual} & ")
                f.write("\\textbf{Mean Conservative} & \\textbf{Impact} & \\textbf{Impact \\%} \\\\\n")
                f.write("\\midrule\n")
                
                quartile_df = subgroup_results['cost_quartile']
                for idx in quartile_df.index:
                    n = int(quartile_df.loc[idx, ('actual', 'count')])
                    mean_actual = quartile_df.loc[idx, ('actual', 'mean')]
                    mean_cons = quartile_df.loc[idx, ('conservative', 'mean')]
                    impact = quartile_df.loc[idx, ('impact', 'sum')]
                    impact_pct = quartile_df.loc[idx, ('impact_pct', 'mean')]
                    
                    f.write(f"{idx} & {n:,} & ")
                    f.write(f"\\${format_number(mean_actual)} & ")
                    f.write(f"\\${format_number(mean_cons)} & ")
                    impact_sign = "+" if impact >= 0 else ""
                    f.write(f"\\${impact_sign}{format_number(impact)} & ")
                    f.write(f"{impact_sign}{format_number(impact_pct, 2)}\\% \\\\\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n\n")
            
            # Impact Level Analysis
            if 'impact_level' in subgroup_results:
                f.write(f"\\begin{{table}}[htbp]\n")
                f.write("\\centering\n")
                f.write("\\small\n")
                f.write(f"\\caption{{Model {model_num}: Distribution by Impact Level \\FiscalYear}}\n")
                f.write(f"\\label{{tab:model{model_num}_impact_distribution}}\n")
                f.write("\\begin{tabular}{lrrrrr}\n")
                f.write("\\toprule\n")
                f.write("\\textbf{Impact Level} & \\textbf{N} & \\textbf{\\%} & ")
                f.write("\\textbf{Mean Actual} & \\textbf{Mean Impact} & \\textbf{Impact \\%} \\\\\n")
                f.write("\\midrule\n")
                
                impact_df = subgroup_results['impact_level']
                total_n = impact_df[('actual', 'count')].sum()
                
                for idx in impact_df.index:
                    n = int(impact_df.loc[idx, ('actual', 'count')])
                    pct = 100 * n / total_n
                    mean_actual = impact_df.loc[idx, ('actual', 'mean')]
                    mean_impact = impact_df.loc[idx, ('impact', 'mean')]
                    impact_pct = impact_df.loc[idx, ('impact_pct', 'mean')]
                    
                    f.write(f"{idx} & {n:,} & {format_number(pct, 1)}\\% & ")
                    f.write(f"\\${format_number(mean_actual)} & ")
                    impact_sign = "+" if mean_impact >= 0 else ""
                    f.write(f"\\${impact_sign}{format_number(mean_impact)} & ")
                    f.write(f"{impact_sign}{format_number(impact_pct, 2)}\\% \\\\\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n\n")
            
            # Figure reference
            f.write(f"Tables~\\ref{{tab:model{model_num}_impact_summary}} through ")
            f.write(f"\\ref{{tab:model{model_num}_impact_distribution}} present detailed ")
            f.write("subgroup analyses, revealing how economic impact varies across ")
            f.write("age groups, living settings, budget levels, and impact categories. ")
            f.write("These breakdowns help identify which populations are most affected ")
            f.write("by prediction errors and where conservative budgeting has the greatest effect.\n\n")
            
            f.write(f"Figure~\\ref{{fig:model{model_num}_impact_histograms}} presents ")
            f.write(f"the distribution analysis for Model {model_num}, showing the distributions ")
            f.write("of actual costs, predicted costs, prediction errors, and conservative ")
            f.write("budget estimates.\n\n")
            
            f.write("\\begin{figure}[htbp]\n")
            f.write("\\centering\n")
            f.write(f"\\includegraphics[width=0.95\\textwidth]{{figures/model_{model_num}_Impact_Histograms.pdf}}\n")
            f.write(f"\\caption{{Model {model_num}: Distribution of costs, predictions, errors, ")
            f.write("and conservative budget estimates. The conservative estimate takes the ")
            f.write("maximum of actual and predicted costs to ensure adequate funding \\FiscalYear.}\n")
            f.write(f"\\label{{fig:model{model_num}_impact_histograms}}\n")
            f.write("\\end{figure}\n\n")
            
            # Interpretation
            if metrics['economic_impact'] > 0:
                f.write(f"The conservative budgeting approach for Model {model_num} would require ")
                f.write(f"an additional \\${format_number(metrics['economic_impact'])} ")
                f.write(f"({format_number(metrics['impact_percentage'], 2)}\\%) compared to actual costs, ")
                f.write(f"averaging \\${format_number(metrics['economic_impact']/metrics['n_samples'])} ")
                f.write("per client. ")
            else:
                f.write(f"The conservative budgeting approach for Model {model_num} would result ")
                f.write(f"in savings of \\${format_number(abs(metrics['economic_impact']))} ")
                f.write(f"({format_number(abs(metrics['impact_percentage']), 2)}\\%) compared to actual costs. ")
            
            f.write(f"The model under-predicted costs in {format_number(metrics['over_budget_pct'], 1)}\\% ")
            f.write("of cases, necessitating the conservative approach to avoid budget shortfalls. ")
            
            # Add insights from subgroup analysis
            if 'impact_level' in subgroup_results:
                impact_df = subgroup_results['impact_level']
                large_increase_idx = 'Large Increase (>25\%)'
                if large_increase_idx in impact_df.index:
                    n_large = int(impact_df.loc[large_increase_idx, ('actual', 'count')])
                    pct_large = 100 * n_large / metrics['n_samples']
                    f.write(f"Notably, {format_number(pct_large, 1)}\\% of cases ({n_large:,} clients) ")
                    f.write("require large budget increases exceeding 25\\%, highlighting the importance ")
                    f.write("of the conservative approach for high-risk cases. ")
            
            f.write("\n\n")
            
            f.write("\\clearpage\n\n")
        
        # Comparative analysis
        f.write(f"\\section{{Comparative Analysis Across Models }}\n")
        f.write("\\label{subsec:comparative_impact}\n\n")
        
        f.write("Table~\\ref{tab:all_models_impact_comparison} presents a comprehensive ")
        f.write("comparison of economic impacts across all budget allocation models.\n\n")
        
        # Comparative table
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\caption{Comparative Economic Impact Analysis Across All Models \\FiscalYear}\n")
        f.write("\\label{tab:all_models_impact_comparison}\n")
        f.write("\\begin{tabular}{lrrrrr}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Model} & \\textbf{Samples} & \\textbf{$R^2$ Test} & ")
        f.write("\\textbf{Economic Impact} & \\textbf{Impact \\%} & \\textbf{Over Budget \\%} \\\\\n")
        f.write("\\midrule\n")
        
        for model_num in MODELS:
            if model_num not in all_results:
                continue
            
            result = all_results[model_num]
            metrics = result['impact_metrics']
            model_metrics = result['model_metrics']
            
            f.write(f"Model {model_num} & ")
            f.write(f"{metrics['n_samples']:,} & ")
            f.write(f"{model_metrics.get('r2_test', 0):.4f} & ")
            impact_sign = "+" if metrics['economic_impact'] >= 0 else ""
            f.write(f"\\${impact_sign}{format_number(metrics['economic_impact'])} & ")
            f.write(f"{impact_sign}{format_number(metrics['impact_percentage'], 2)}\\% & ")
            f.write(f"{format_number(metrics['over_budget_pct'], 1)}\\% \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Summary insights
        f.write("\\section{Key Insights}\n\n")
        f.write("\\begin{itemize}\n")
        
        # Find best and worst performing models
        best_r2_model = max(all_results.keys(), 
                           key=lambda x: all_results[x]['model_metrics'].get('r2_test', 0))
        worst_impact_pct_model = max(all_results.keys(),
                                     key=lambda x: all_results[x]['impact_metrics']['impact_percentage'])
        
        f.write(f"\\item Model {best_r2_model} achieves the highest predictive accuracy ")
        f.write(f"with $R^2$ = {all_results[best_r2_model]['model_metrics'].get('r2_test', 0):.4f}.\n")
        
        f.write(f"\\item Model {worst_impact_pct_model} requires the largest conservative ")
        f.write(f"budget adjustment at {format_number(all_results[worst_impact_pct_model]['impact_metrics']['impact_percentage'], 2)}\\%.\n")
        
        f.write("\\item The conservative budgeting approach ensures adequate funding to cover ")
        f.write("cases where the model under-predicts actual costs.\n")
        
        f.write("\\item Economic impact percentages reflect both model accuracy and the ")
        f.write("degree of systematic under- or over-prediction.\n")
        
        f.write("\\item Subgroup analyses reveal differential impacts across age groups, living ")
        f.write("settings, and budget levels, providing insights for targeted policy interventions.\n")
        
        f.write("\\item Impact level distributions identify high-risk cases requiring substantial ")
        f.write("budget adjustments beyond model predictions.\n")
        
        f.write("\\end{itemize}\n\n")
        
    print(f"LaTeX report generated: {output_file}")
    return output_file


def main():
    """Main execution function."""
    print("="*70)
    print("Economic Impact Analysis Report Generator")
    print("="*70)
    print()
    
    all_results = {}
    
    for model_num in MODELS:
        print(f"Processing Model {model_num}...")
        
        try:
            # Load data
            predictions_df, model_metrics = load_model_data(model_num,  BASE_YEAR)
            print(f"  Loaded {len(predictions_df)} predictions")
            
            # Calculate conservative estimates
            conservative = calculate_conservative_estimates(predictions_df)
            
            # Calculate impact metrics
            impact_metrics = calculate_impact_metrics(predictions_df, conservative)
            print(f"  Economic Impact: ${impact_metrics['economic_impact']:,.2f} "
                  f"({impact_metrics['impact_percentage']:.2f}%)")
            
            # Analyze subgroups
            subgroup_results = analyze_subgroups(predictions_df, conservative)
            print(f"  Subgroup analysis complete")
            
            # Create histograms
            fig_path = create_histograms(predictions_df, conservative, model_num)
            print(f"  Generated: {fig_path.name}")
            
            # Store results
            all_results[model_num] = {
                'predictions': predictions_df,
                'conservative': conservative,
                'impact_metrics': impact_metrics,
                'model_metrics': model_metrics,
                'subgroup_results': subgroup_results,
                'figure_path': fig_path
            }
            
            print()
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            print()
            continue
    
    # Generate LaTeX output
    if all_results:
        print("Generating LaTeX report...")
        latex_file = generate_latex_output(all_results)
        print()
        print("="*70)
        print(f"Analysis complete! Generated {len(all_results)} model reports.")
        print(f"LaTeX output: {latex_file}")
        print(f"Figures saved to: {FIGURES_DIR}")
        print("="*70)
    else:
        print("ERROR: No models were successfully processed.")


if __name__ == "__main__":
    main()