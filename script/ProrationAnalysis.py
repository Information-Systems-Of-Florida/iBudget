"""
Monthly Cost Distribution Analysis for Proration Assessment - Complete Version
This script analyzes how costs are distributed across months for each customer
to determine if proration would be a valid approach for mid-year QSI changes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

class CostDistributionAnalyzer:
    def __init__(self):
        """Initialize the analyzer"""
        self.consumer_months = None
        self.metrics = None
        self.figures_dir = Path('../report/figures')
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
    def get_connection_string(self):
        """Get the connection string for SQL Server"""
        # Configuration
        server = '.'  # localhost
        database = 'APD'
        
        # For Windows Authentication
        username = None  
        password = None
        
        if username and password:
            connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
        else:
            # Windows Authentication
            connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes'
        
        return connection_string
        
    def load_monthly_costs(self):
        """Load monthly cost data for all consumers"""
        
        print("Loading monthly cost data from database...")
        
        query = """
        WITH MonthlyData AS (
            SELECT 
                CaseNo,
                YEAR(ServiceDate) as ServiceYear,
                MONTH(ServiceDate) as ServiceMonth,
                -- Assign fiscal year
                CASE 
                    WHEN MONTH(ServiceDate) >= 9 
                    THEN YEAR(ServiceDate)
                    ELSE YEAR(ServiceDate) - 1
                END AS FiscalYear,
                -- Assign fiscal month (1-12 where 1 = September)
                CASE 
                    WHEN MONTH(ServiceDate) >= 9 
                    THEN MONTH(ServiceDate) - 8
                    ELSE MONTH(ServiceDate) + 4
                END AS FiscalMonth,
                SUM(PaidAmt) as MonthlyPaidAmt
            FROM tbl_Claims_MMIS
            WHERE PaidAmt > 0  -- Exclude negative and zero amounts
            GROUP BY 
                CaseNo,
                YEAR(ServiceDate),
                MONTH(ServiceDate)
        )
        SELECT 
            CaseNo,
            FiscalYear,
            FiscalMonth,
            MonthlyPaidAmt
        FROM MonthlyData
        ORDER BY CaseNo, FiscalYear, FiscalMonth
        """
        
        connection_string = self.get_connection_string()
        
        with pyodbc.connect(connection_string) as conn:
            self.consumer_months = pd.read_sql(query, conn)
            
        # Convert decimal types to float if needed
        self.consumer_months['MonthlyPaidAmt'] = pd.to_numeric(
            self.consumer_months['MonthlyPaidAmt'], errors='coerce'
        )
        
        print(f"Loaded {len(self.consumer_months)} monthly records")
        print(f"Covering {self.consumer_months['CaseNo'].nunique()} unique customers")
        
    def calculate_metrics(self):
        """Calculate q and r metrics for each consumer-fiscal year"""
        
        print("Calculating distribution metrics...")
        
        metrics_list = []
        
        # Group by consumer and fiscal year
        grouped = self.consumer_months.groupby(['CaseNo', 'FiscalYear'])
        
        for (case_no, fiscal_year), group_df in grouped:
            # Make a copy to avoid modifying the original
            group = group_df.copy()
            
            # Count months with services
            months_with_service = len(group)
            
            # Calculate q = months with services / 12
            q = months_with_service / 12.0
            
            # Get monthly costs indexed by fiscal month
            monthly_costs_dict = dict(zip(group['FiscalMonth'], group['MonthlyPaidAmt']))
            
            # Create full year series with zeros for missing months
            all_months = pd.Series(
                [monthly_costs_dict.get(i, 0.0) for i in range(1, 13)],
                index=range(1, 13),
                dtype=float
            )
            
            # Get only the months with actual costs for variance calculation
            actual_costs = group['MonthlyPaidAmt'].values
            
            # Calculate metrics
            if len(actual_costs) > 0 and actual_costs.sum() > 0:
                # Average of months with actual costs
                a = actual_costs.mean()
                
                # Calculate r = sum((m_i - a)^2 / a) for months with costs
                if a > 0:
                    r = np.sum((actual_costs - a) ** 2 / a)
                else:
                    r = 0
                    
                # Coefficient of variation
                cv = actual_costs.std() / a if a > 0 else 0
            else:
                a = 0
                r = 0
                cv = 0
            
            # Store metrics
            metrics_list.append({
                'CaseNo': case_no,
                'FiscalYear': fiscal_year,
                'q_coverage': q,
                'r_variance': r,
                'months_with_service': months_with_service,
                'total_annual_cost': actual_costs.sum() if len(actual_costs) > 0 else 0,
                'mean_monthly_cost': a,
                'coefficient_variation': cv,
                'has_full_year': months_with_service == 12
            })
        
        self.metrics = pd.DataFrame(metrics_list)
        
        # Clean up extreme values for visualization
        # Cap r at 99th percentile for better visualization
        if len(self.metrics) > 0:
            r_99 = self.metrics['r_variance'].quantile(0.99)
            self.metrics['r_variance_capped'] = self.metrics['r_variance'].clip(upper=r_99)
        else:
            self.metrics['r_variance_capped'] = self.metrics['r_variance']
        
        print(f"Calculated metrics for {len(self.metrics)} customer-years")
        
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        
        print("Generating visualizations...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Main 2D histogram (q vs r)
        ax1 = plt.subplot(2, 3, 1)
        h = ax1.hist2d(self.metrics['q_coverage'], 
                       self.metrics['r_variance_capped'],
                       bins=[12, 50],
                       cmap='YlOrRd',
                       cmin=1)
        plt.colorbar(h[3], ax=ax1, label='Count')
        ax1.set_xlabel('q (Coverage: Months with Service / 12)')
        ax1.set_ylabel('r (Variance: Σ((m_i - a)²/a))')
        ax1.set_title('Distribution of Coverage vs Variance')
        ax1.grid(True, alpha=0.3)
        
        # 2. Marginal distribution of q
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(self.metrics['q_coverage'], bins=12, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('q (Coverage: Months with Service / 12)')
        ax2.set_ylabel('Count of Customer-Years')
        ax2.set_title('Distribution of Monthly Coverage')
        ax2.axvline(x=1.0, color='r', linestyle='--', label='Full Year')
        ax2.axvline(x=0.75, color='orange', linestyle='--', label='9 months')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add text with statistics
        full_year_pct = (self.metrics['q_coverage'] == 1.0).mean() * 100
        nine_plus_pct = (self.metrics['q_coverage'] >= 0.75).mean() * 100
        ax2.text(0.5, 0.95, f'Full year: {full_year_pct:.1f}%\n≥9 months: {nine_plus_pct:.1f}%',
                transform=ax2.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Marginal distribution of r
        ax3 = plt.subplot(2, 3, 3)
        # Use log scale for better visualization
        r_positive = self.metrics[self.metrics['r_variance'] > 0]['r_variance']
        ax3.hist(np.log10(r_positive + 1), bins=50, edgecolor='black', alpha=0.7)
        ax3.set_xlabel('log10(r + 1) (Log of Variance Metric)')
        ax3.set_ylabel('Count of Customer-Years')
        ax3.set_title('Distribution of Cost Variance')
        ax3.grid(True, alpha=0.3)
        
        # Add median and percentiles
        median_r = self.metrics['r_variance'].median()
        p75_r = self.metrics['r_variance'].quantile(0.75)
        ax3.text(0.95, 0.95, f'Median r: {median_r:.1f}\n75th %ile: {p75_r:.1f}',
                transform=ax3.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. Scatter plot colored by total cost
        ax4 = plt.subplot(2, 3, 4)
        scatter = ax4.scatter(self.metrics['q_coverage'],
                             self.metrics['r_variance_capped'],
                             c=np.log10(self.metrics['total_annual_cost'] + 1),
                             alpha=0.5,
                             s=10,
                             cmap='viridis')
        plt.colorbar(scatter, ax=ax4, label='log10(Annual Cost)')
        ax4.set_xlabel('q (Coverage)')
        ax4.set_ylabel('r (Variance)')
        ax4.set_title('Coverage vs Variance (colored by cost)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Box plot of r by coverage categories
        ax5 = plt.subplot(2, 3, 5)
        coverage_cats = pd.cut(self.metrics['q_coverage'], 
                               bins=[0, 0.25, 0.5, 0.75, 0.99, 1.0],
                               labels=['≤3mo', '4-6mo', '7-9mo', '10-11mo', '12mo'])
        self.metrics['coverage_category'] = coverage_cats
        
        # Filter out extreme r values for better visualization
        r_filtered = self.metrics[self.metrics['r_variance'] < self.metrics['r_variance'].quantile(0.95)]
        
        r_filtered.boxplot(column='r_variance', by='coverage_category', ax=ax5)
        ax5.set_xlabel('Coverage Category')
        ax5.set_ylabel('r (Variance Metric)')
        ax5.set_title('Variance by Coverage Level')
        plt.sca(ax5)
        plt.xticks(rotation=45)
        
        # 6. Summary statistics table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Calculate summary stats
        summary_stats = []
        summary_stats.append(['Total Customer-Years', f"{len(self.metrics):,}"])
        summary_stats.append(['Full Year Coverage (q=1)', f"{(self.metrics['q_coverage'] == 1.0).sum():,} ({full_year_pct:.1f}%)"])
        summary_stats.append(['≥9 Months (q≥0.75)', f"{(self.metrics['q_coverage'] >= 0.75).sum():,} ({nine_plus_pct:.1f}%)"])
        summary_stats.append(['<6 Months (q<0.5)', f"{(self.metrics['q_coverage'] < 0.5).sum():,} ({(self.metrics['q_coverage'] < 0.5).mean()*100:.1f}%)"])
        summary_stats.append(['', ''])
        summary_stats.append(['Mean q', f"{self.metrics['q_coverage'].mean():.3f}"])
        summary_stats.append(['Median q', f"{self.metrics['q_coverage'].median():.3f}"])
        summary_stats.append(['Mean r', f"{self.metrics['r_variance'].mean():.1f}"])
        summary_stats.append(['Median r', f"{self.metrics['r_variance'].median():.1f}"])
        
        table = ax6.table(cellText=summary_stats,
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax6.set_title('Summary Statistics', pad=20, fontsize=12, fontweight='bold')
        
        plt.suptitle('Monthly Cost Distribution Analysis for Proration Assessment', fontsize=14, y=1.02)
        plt.tight_layout()
        
        # Save figure
        output_path = self.figures_dir / 'cost_distribution_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_path}")
        
    def analyze_proration_feasibility(self):
        """Analyze and report on proration feasibility"""
        
        print("\n" + "="*60)
        print("PRORATION FEASIBILITY ANALYSIS")
        print("="*60)
        
        # Analyze different segments
        full_year = self.metrics[self.metrics['q_coverage'] == 1.0]
        high_coverage = self.metrics[self.metrics['q_coverage'] >= 0.75]
        mid_coverage = self.metrics[(self.metrics['q_coverage'] >= 0.5) & (self.metrics['q_coverage'] < 0.75)]
        low_coverage = self.metrics[self.metrics['q_coverage'] < 0.5]
        
        print(f"\nCoverage Distribution:")
        print(f"  Full year (q=1.0): {len(full_year):,} ({len(full_year)/len(self.metrics)*100:.1f}%)")
        print(f"  High (q≥0.75): {len(high_coverage):,} ({len(high_coverage)/len(self.metrics)*100:.1f}%)")
        print(f"  Medium (0.5≤q<0.75): {len(mid_coverage):,} ({len(mid_coverage)/len(self.metrics)*100:.1f}%)")
        print(f"  Low (q<0.5): {len(low_coverage):,} ({len(low_coverage)/len(self.metrics)*100:.1f}%)")
        
        print(f"\nVariance Analysis (r metric):")
        print(f"  Median r for full year: {full_year['r_variance'].median():.2f}")
        print(f"  Median r for high coverage: {high_coverage['r_variance'].median():.2f}")
        print(f"  Median r for medium coverage: {mid_coverage['r_variance'].median():.2f}")
        print(f"  Median r for low coverage: {low_coverage['r_variance'].median():.2f}")
        
        # Analyze consistency
        low_variance_threshold = self.metrics['r_variance'].quantile(0.5)
        consistent_spenders = self.metrics[self.metrics['r_variance'] < low_variance_threshold]
        
        print(f"\nConsistency Analysis:")
        print(f"  Customer-years with below-median variance: {len(consistent_spenders):,} ({len(consistent_spenders)/len(self.metrics)*100:.1f}%)")
        print(f"  Of these, % with full coverage: {(consistent_spenders['q_coverage']==1.0).mean()*100:.1f}%")
        print(f"  Of these, % with ≥9 months: {(consistent_spenders['q_coverage']>=0.75).mean()*100:.1f}%")
        
        # Save detailed results
        self.metrics.to_csv(self.figures_dir / 'cost_distribution_metrics.csv', index=False)
        print(f"\nDetailed metrics saved to cost_distribution_metrics.csv")
        
    def generate_latex_output(self):
        """Generate LaTeX commands and subsection for the report"""
        
        print("\nGenerating LaTeX output files...")
        
        # Calculate summary statistics for LaTeX
        full_year = self.metrics[self.metrics['q_coverage'] == 1.0]
        high_coverage = self.metrics[self.metrics['q_coverage'] >= 0.75]
        low_coverage = self.metrics[self.metrics['q_coverage'] < 0.5]
        
        full_year_pct = len(full_year) / len(self.metrics) * 100
        high_coverage_pct = len(high_coverage) / len(self.metrics) * 100
        median_r_full = full_year['r_variance'].median()
        median_r_low = low_coverage['r_variance'].median()
        
        # Generate LaTeX commands file
        commands_content = f"""% Proration Analysis Statistics
% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

% Coverage statistics
\\renewcommand{{\\ProrationFullYearPct}}{{{full_year_pct:.1f}}}
\\renewcommand{{\\ProrationHighCoveragePct}}{{{high_coverage_pct:.1f}}}
\\renewcommand{{\\ProrationMedianRFull}}{{{median_r_full:.2f}}}
\\renewcommand{{\\ProrationMedianRLow}}{{{median_r_low:.2f}}}

% Customer counts
\\renewcommand{{\\ProrationTotalCustomerYears}}{{{len(self.metrics):,}}}
\\renewcommand{{\\ProrationFullYearCount}}{{{len(full_year):,}}}
\\renewcommand{{\\ProrationHighCoverageCount}}{{{len(high_coverage):,}}}
"""
        
        # Save commands file
        commands_path = self.figures_dir.parent / 'proration_commands.tex'
        with open(commands_path, 'w') as f:
            f.write(commands_content)
        
        print(f"LaTeX commands saved to {commands_path}")
        
        # Generate the subsection content
        subsection_content = r"""\subsection{Proration Analysis and Decision}

Given the substantial data loss from excluding mid-year QSI changes, we investigated whether costs could be prorated for partial-year records. The analysis examined monthly cost distributions for all customers, calculating two metrics:

\begin{align}
q &= \frac{\text{months with services}}{12} \\
r &= \sum_{i=1}^{12} \frac{(m_i - \bar{m})^2}{\bar{m}}
\end{align}

where $m_i$ represents monthly expenses and $\bar{m}$ is the average monthly expense for that customer-year.

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{figures/cost_distribution_analysis.png}
    \caption{Monthly cost distribution analysis for proration assessment}
    \label{fig:proration_analysis}
\end{figure}

Figure~\ref{fig:proration_analysis} presents the comprehensive proration feasibility analysis:

\begin{itemize}
    \item \textbf{Top-left panel}: Two-dimensional histogram showing the relationship between coverage ($q$) and variance ($r$). The concentration of points at $q=1$ demonstrates that most customer-years have complete coverage, while the vertical spread indicates substantial variance in spending patterns.
    
    \item \textbf{Top-center panel}: Distribution of monthly coverage showing a bimodal pattern---customers either have full-year coverage or very limited engagement, with few in between.
    
    \item \textbf{Top-right panel}: Log-scale distribution of the variance metric revealing that cost variance spans multiple orders of magnitude, indicating highly heterogeneous spending patterns.
    
    \item \textbf{Bottom-left panel}: Scatter plot colored by annual cost magnitude shows that higher-cost customers tend to have more complete coverage but also higher variance.
    
    \item \textbf{Bottom-center panel}: Box plots of variance by coverage category reveal the counterintuitive finding that full-year customers have the highest cost variance.
    
    \item \textbf{Bottom-right panel}: Summary statistics quantifying the distribution patterns.
\end{itemize}

The analysis revealed critical findings:
\begin{itemize}
    \item \textbf{\ProrationFullYearPct\% of customer-years} have full 12-month coverage ($q = 1.0$)
    \item \textbf{Median variance for full-year customers}: $r = \ProrationMedianRFull$
    \item \textbf{Median variance for low-coverage customers}: $r = \ProrationMedianRLow$
\end{itemize}

The counterintuitively high variance (\ProrationMedianRFull) for full-year customers indicates highly irregular spending patterns---likely reflecting equipment purchases, hospitalizations, or seasonal service variations. This ``lumpy'' cost distribution makes proration inadvisable for several reasons:

\begin{enumerate}
    \item \textbf{Attribution ambiguity}: With such high monthly variance, costs cannot be fairly attributed to different QSI assessment periods when changes occur mid-year.
    
    \item \textbf{Systematic bias}: The inverse relationship between coverage and variance suggests partial-year customers represent a fundamentally different population that would bias model calibration.
    
    \item \textbf{Implementation complexity}: Any proration scheme would require sophisticated adjustment factors that vary by customer characteristics, introducing additional uncertainty.
\end{enumerate}

Consequently, we maintain the conservative exclusion strategy despite the data loss, as the integrity of the QSI-to-cost relationship is paramount for regulatory compliance and fair budget allocation.
"""
        
        # Save subsection file
        subsection_path = self.figures_dir.parent / 'proration_analysis.tex'
        with open(subsection_path, 'w') as f:
            f.write(subsection_content)
        
        print(f"LaTeX subsection saved to {subsection_path}")
        
        # Generate list of commands for master config
        print("\n" + "="*60)
        print("Commands to add to master configuration file:")
        print("="*60)
        master_commands = r"""% Proration Analysis Commands (add to master config)
\newcommand{\ProrationFullYearPct}{\WarningRunPipeline}
\newcommand{\ProrationHighCoveragePct}{\WarningRunPipeline}
\newcommand{\ProrationMedianRFull}{\WarningRunPipeline}
\newcommand{\ProrationMedianRLow}{\WarningRunPipeline}
\newcommand{\ProrationTotalCustomerYears}{\WarningRunPipeline}
\newcommand{\ProrationFullYearCount}{\WarningRunPipeline}
\newcommand{\ProrationHighCoverageCount}{\WarningRunPipeline}"""
        
        print(master_commands)
        
        # Save to file for easy copying
        master_commands_path = self.figures_dir.parent / 'proration_master_commands.txt'
        with open(master_commands_path, 'w') as f:
            f.write(master_commands)
        
        print(f"\nMaster commands also saved to: {master_commands_path}")
        print("="*60)
        
    def run_analysis(self):
        """Execute the complete analysis"""
        
        print("Starting Monthly Cost Distribution Analysis")
        print("="*60)
        
        # Load data
        self.load_monthly_costs()
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Analyze proration feasibility
        self.analyze_proration_feasibility()
        
        # Generate LaTeX output
        self.generate_latex_output()
        
        print("\n" + "="*60)
        print("Analysis Complete!")
        print("="*60)


def main():
    """Main entry point"""
    
    # Create analyzer
    analyzer = CostDistributionAnalyzer()
    
    # Run analysis
    analyzer.run_analysis()


if __name__ == "__main__":
    main()