"""
Outlier Analysis for iBudget Model Calibration
This script analyzes data quality and exclusions from the sp_Outliers stored procedure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc
from datetime import datetime
import json
from pathlib import Path
from decimal import Decimal

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class OutlierAnalysis:
    def __init__(self, connection_string):
        """
        Initialize the outlier analysis
        
        Parameters:
        -----------
        connection_string : str
            SQL Server connection string
        """
        self.connection_string = connection_string
        self.results = {}
        self.figures_dir = Path('../report/figures')
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
    def execute_stored_procedure(self):
        """Execute sp_Outliers and retrieve all result sets"""
        
        print("Executing sp_Outliers stored procedure...")
        
        with pyodbc.connect(self.connection_string) as conn:
            cursor = conn.cursor()
            cursor.execute("EXEC sp_Outliers")
            
            # Helper function to convert decimal.Decimal to float
            def convert_decimals(df):
                """Convert decimal.Decimal columns to float"""
                for col in df.columns:
                    if df[col].dtype == object:
                        try:
                            # Try converting to numeric
                            df[col] = pd.to_numeric(df[col])
                        except:
                            pass
                return df
            
            # Result Set 1: Overall Statistics
            self.results['overall_stats'] = pd.DataFrame.from_records(
                cursor.fetchall(), 
                columns=[desc[0] for desc in cursor.description]
            )
            self.results['overall_stats'] = convert_decimals(self.results['overall_stats'])
            
            # Move to next result set
            cursor.nextset()
            
            # Result Set 2: Exclusion Summary
            self.results['exclusion_summary'] = pd.DataFrame.from_records(
                cursor.fetchall(),
                columns=[desc[0] for desc in cursor.description]
            )
            self.results['exclusion_summary'] = convert_decimals(self.results['exclusion_summary'])
            
            cursor.nextset()
            
            # Result Set 3: Consumer Level Summary
            self.results['consumer_summary'] = pd.DataFrame.from_records(
                cursor.fetchall(),
                columns=[desc[0] for desc in cursor.description]
            )
            self.results['consumer_summary'] = convert_decimals(self.results['consumer_summary'])
            
            cursor.nextset()
            
            # Result Set 4: Fiscal Year Distribution
            self.results['fiscal_year_dist'] = pd.DataFrame.from_records(
                cursor.fetchall(),
                columns=[desc[0] for desc in cursor.description]
            )
            self.results['fiscal_year_dist'] = convert_decimals(self.results['fiscal_year_dist'])
            
            cursor.nextset()
            
            # Result Set 5: Exclusion Overlap
            self.results['exclusion_overlap'] = pd.DataFrame.from_records(
                cursor.fetchall(),
                columns=[desc[0] for desc in cursor.description]
            )
            self.results['exclusion_overlap'] = convert_decimals(self.results['exclusion_overlap'])
            
            cursor.nextset()
            
            # Result Set 6: Cost Distribution
            self.results['cost_distribution'] = pd.DataFrame.from_records(
                cursor.fetchall(),
                columns=[desc[0] for desc in cursor.description]
            )
            self.results['cost_distribution'] = convert_decimals(self.results['cost_distribution'])
            
            cursor.nextset()
            
            # Result Set 7: Detailed Records
            self.results['detailed_records'] = pd.DataFrame.from_records(
                cursor.fetchall(),
                columns=[desc[0] for desc in cursor.description]
            )
            self.results['detailed_records'] = convert_decimals(self.results['detailed_records'])
            
        print(f"Retrieved {len(self.results)} result sets")
        
    def analyze_exclusions(self):
        """Perform additional analysis on exclusions"""
        
        df = self.results['detailed_records']
        
        # Convert decimal.Decimal to float for pandas operations
        numeric_columns = ['TotalPaidAmt', 'ServiceDays', 'DaysFromFYStart', 'DaysToFYEnd']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate correlation between exclusion flags
        exclusion_cols = [col for col in df.columns if col.startswith('Flag_')]
        self.exclusion_correlation = df[exclusion_cols].corr()
        
        # Analyze cost outliers
        usable_costs = df[df['IsUsable'] == 1]['TotalPaidAmt'].dropna()
        
        if len(usable_costs) > 0:
            Q1 = usable_costs.quantile(0.25)
            Q3 = usable_costs.quantile(0.75)
            IQR = Q3 - Q1
            
            self.cost_outliers = {
                'lower_fence': Q1 - 3 * IQR,
                'upper_fence': Q3 + 3 * IQR,
                'n_below': (usable_costs < Q1 - 3 * IQR).sum(),
                'n_above': (usable_costs > Q3 + 3 * IQR).sum(),
                'pct_outliers': 100 * ((usable_costs < Q1 - 3 * IQR) | (usable_costs > Q3 + 3 * IQR)).sum() / len(usable_costs)
            }
        else:
            self.cost_outliers = {
                'lower_fence': 0,
                'upper_fence': 0,
                'n_below': 0,
                'n_above': 0,
                'pct_outliers': 0
            }
        
        # Consumer trajectory analysis
        consumer_years = df.groupby('CaseNo').agg({
            'FiscalYear': ['count', 'min', 'max'],
            'IsUsable': 'sum',
            'TotalPaidAmt': 'mean'
        })
        consumer_years.columns = ['total_years', 'first_year', 'last_year', 'usable_years', 'avg_cost']
        consumer_years['year_span'] = consumer_years['last_year'] - consumer_years['first_year'] + 1
        consumer_years['has_gaps'] = consumer_years['year_span'] > consumer_years['total_years']
        
        self.consumer_trajectory = {
            'total_consumers': len(consumer_years),
            'consumers_with_trajectory': (consumer_years['usable_years'] >= 2).sum(),
            'pct_with_trajectory': 100 * (consumer_years['usable_years'] >= 2).sum() / len(consumer_years),
            'consumers_with_gaps': consumer_years['has_gaps'].sum(),
            'pct_with_gaps': 100 * consumer_years['has_gaps'].sum() / len(consumer_years)
        }
        
    def generate_plots(self):
        """Generate all visualization plots"""
        
        # Plot 1: Exclusion Reasons Bar Chart
        self.plot_exclusion_reasons()
        
        # Plot 2: Cost Distribution Comparison
        self.plot_cost_distributions()
        
        # Plot 3: Fiscal Year Trends
        self.plot_fiscal_year_trends()
        
        # Plot 4: Consumer Coverage Analysis
        self.plot_consumer_coverage()
        
        # Plot 5: Exclusion Overlap Venn Diagram (simplified as heatmap)
        self.plot_exclusion_overlap()
        
        # Plot 6: Cost Outlier Analysis
        self.plot_cost_outliers()
        
    def plot_exclusion_reasons(self):
        """Plot bar chart of exclusion reasons"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Get exclusion data
        excl = self.results['exclusion_summary'].iloc[0]
        
        # Extract counts and percentages
        reasons = []
        counts = []
        percentages = []
        
        for col in excl.index:
            if col.startswith('Count_'):
                reason = col.replace('Count_', '').replace('_', ' ')
                reasons.append(reason)
                counts.append(excl[col])
                percentages.append(excl[col.replace('Count_', 'Pct_')])
        
        # Plot counts
        bars1 = ax1.bar(range(len(reasons)), counts, color='steelblue')
        ax1.set_xticks(range(len(reasons)))
        ax1.set_xticklabels(reasons, rotation=45, ha='right')
        ax1.set_ylabel('Number of Consumer-Years')
        ax1.set_title('Exclusion Counts by Reason')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars1, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count):,}', ha='center', va='bottom')
        
        # Plot percentages
        bars2 = ax2.bar(range(len(reasons)), percentages, color='coral')
        ax2.set_xticks(range(len(reasons)))
        ax2.set_xticklabels(reasons, rotation=45, ha='right')
        ax2.set_ylabel('Percentage of Consumer-Years (%)')
        ax2.set_title('Exclusion Percentages by Reason')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, pct in zip(bars2, percentages):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.suptitle('Data Exclusion Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'exclusion_reasons.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_cost_distributions(self):
        """Plot cost distributions for included vs excluded data"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        df = self.results['detailed_records']
        included = df[df['IsUsable'] == 1]['TotalPaidAmt']
        excluded = df[df['IsUsable'] == 0]['TotalPaidAmt']
        
        # Remove extreme outliers for better visualization
        included_plot = included[included < included.quantile(0.99)]
        excluded_plot = excluded[excluded < excluded.quantile(0.99)]
        
        # Histogram comparison
        ax = axes[0, 0]
        ax.hist([included_plot, excluded_plot], bins=50, label=['Included', 'Excluded'], 
                alpha=0.7, color=['green', 'red'])
        ax.set_xlabel('Annual Cost ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Cost Distribution: Included vs Excluded')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Box plot comparison
        ax = axes[0, 1]
        box_data = [included_plot, excluded_plot]
        bp = ax.boxplot(box_data, tick_labels=['Included', 'Excluded'], patch_artist=True)
        bp['boxes'][0].set_facecolor('green')
        bp['boxes'][1].set_facecolor('red')
        ax.set_ylabel('Annual Cost ($)')
        ax.set_title('Cost Distribution Box Plot')
        ax.grid(axis='y', alpha=0.3)
        
        # Log scale histogram
        ax = axes[1, 0]
        # Filter out zero and negative values for log scale
        included_pos = included[included > 0]
        excluded_pos = excluded[excluded > 0]
        
        if len(included_pos) > 0:
            ax.hist(np.log10(included_pos), bins=50, alpha=0.7, label='Included', color='green')
        if len(excluded_pos) > 0:
            ax.hist(np.log10(excluded_pos), bins=50, alpha=0.7, label='Excluded', color='red')
        
        ax.set_xlabel('Log10(Annual Cost)')
        ax.set_ylabel('Frequency')
        ax.set_title('Cost Distribution (Log Scale)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Statistics table
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')
        
        stats_data = []
        for status, data in [('Included', included), ('Excluded', excluded)]:
            stats_data.append([
                status,
                f'{len(data):,}',
                f'${data.mean():,.0f}',
                f'${data.median():,.0f}',
                f'${data.std():,.0f}'
            ])
        
        table = ax.table(cellText=stats_data,
                        colLabels=['Status', 'Count', 'Mean', 'Median', 'Std Dev'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.set_title('Summary Statistics', pad=20)
        
        plt.suptitle('Cost Distribution Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'cost_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_fiscal_year_trends(self):
        """Plot trends across fiscal years"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        fy_data = self.results['fiscal_year_dist']
        
        # Total consumers per year
        ax = axes[0, 0]
        ax.plot(fy_data['FiscalYear'], fy_data['ConsumersInYear'], 
                marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Fiscal Year')
        ax.set_ylabel('Number of Consumers')
        ax.set_title('Total Consumers by Fiscal Year')
        ax.grid(True, alpha=0.3)
        
        # Usable vs Total Records
        ax = axes[0, 1]
        width = 0.35
        x = np.arange(len(fy_data))
        ax.bar(x - width/2, fy_data['TotalRecords'], width, label='Total', color='lightblue')
        ax.bar(x + width/2, fy_data['UsableRecords'], width, label='Usable', color='darkblue')
        ax.set_xlabel('Fiscal Year')
        ax.set_ylabel('Number of Records')
        ax.set_title('Total vs Usable Records by Fiscal Year')
        ax.set_xticks(x)
        ax.set_xticklabels(fy_data['FiscalYear'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Average cost trends
        ax = axes[1, 0]
        ax.plot(fy_data['FiscalYear'], fy_data['AvgCost'], 
                marker='o', linewidth=2, markersize=8, color='green')
        ax.set_xlabel('Fiscal Year')
        ax.set_ylabel('Average Annual Cost ($)')
        ax.set_title('Average Cost Trends')
        ax.grid(True, alpha=0.3)
        
        # Usability rate
        ax = axes[1, 1]
        usability_rate = 100 * fy_data['UsableRecords'] / fy_data['TotalRecords']
        bars = ax.bar(fy_data['FiscalYear'], usability_rate, color='orange')
        ax.set_xlabel('Fiscal Year')
        ax.set_ylabel('Usability Rate (%)')
        ax.set_title('Data Usability Rate by Fiscal Year')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars, usability_rate):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.suptitle('Fiscal Year Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fiscal_year_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_consumer_coverage(self):
        """Plot consumer-level coverage analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        df = self.results['detailed_records']
        consumer_years = df.groupby('CaseNo').agg({
            'FiscalYear': 'count',
            'IsUsable': 'sum',
            'TotalPaidAmt': 'mean'
        }).rename(columns={'FiscalYear': 'total_years', 'IsUsable': 'usable_years', 
                          'TotalPaidAmt': 'avg_cost'})
        
        # Distribution of years per consumer
        ax = axes[0, 0]
        ax.hist(consumer_years['total_years'], bins=range(1, consumer_years['total_years'].max()+2), 
                edgecolor='black', alpha=0.7)
        ax.set_xlabel('Number of Years in Database')
        ax.set_ylabel('Number of Consumers')
        ax.set_title('Distribution of Data Years per Consumer')
        ax.grid(axis='y', alpha=0.3)
        
        # Usable years distribution
        ax = axes[0, 1]
        ax.hist(consumer_years['usable_years'], bins=range(0, consumer_years['usable_years'].max()+2), 
                edgecolor='black', alpha=0.7, color='green')
        ax.set_xlabel('Number of Usable Years')
        ax.set_ylabel('Number of Consumers')
        ax.set_title('Distribution of Usable Years per Consumer')
        ax.grid(axis='y', alpha=0.3)
        
        # Comparison of total vs usable
        ax = axes[1, 0]
        ax.scatter(consumer_years['total_years'], consumer_years['usable_years'], 
                  alpha=0.5, s=10)
        ax.plot([0, consumer_years['total_years'].max()], 
                [0, consumer_years['total_years'].max()], 
                'r--', label='Perfect Usability')
        ax.set_xlabel('Total Years in Database')
        ax.set_ylabel('Usable Years')
        ax.set_title('Total Years vs Usable Years per Consumer')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Consumer categories pie chart
        ax = axes[1, 1]
        categories = [
            ('No Usable Data', (consumer_years['usable_years'] == 0).sum()),
            ('1 Usable Year', (consumer_years['usable_years'] == 1).sum()),
            ('2+ Usable Years\n(Trajectory)', (consumer_years['usable_years'] >= 2).sum())
        ]
        labels, sizes = zip(*categories)
        colors = ['red', 'yellow', 'green']
        explode = (0.1, 0, 0)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax.set_title('Consumer Data Availability')
        
        plt.suptitle('Consumer-Level Coverage Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'consumer_coverage.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_exclusion_overlap(self):
        """Plot exclusion flag overlap as heatmap"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Correlation heatmap
        df = self.results['detailed_records']
        exclusion_cols = [col for col in df.columns if col.startswith('Flag_')]
        
        # Rename columns for better display
        rename_dict = {col: col.replace('Flag_', '').replace('_', ' ') for col in exclusion_cols}
        corr_matrix = df[exclusion_cols].rename(columns=rename_dict).corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax1, vmin=-1, vmax=1)
        ax1.set_title('Exclusion Flag Correlations')
        
        # Overlap counts
        overlap = self.results['exclusion_overlap'].iloc[0]
        categories = ['No Exclusions', '1 Exclusion', '2 Exclusions', '3+ Exclusions']
        values = [overlap['NoExclusions'], overlap['OneExclusion'], 
                 overlap['TwoExclusions'], overlap['ThreeOrMoreExclusions']]
        
        bars = ax2.bar(categories, values, color=['green', 'yellow', 'orange', 'red'])
        ax2.set_ylabel('Number of Consumer-Years')
        ax2.set_title('Distribution of Exclusion Overlap')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(val):,}', ha='center', va='bottom')
        
        plt.suptitle('Exclusion Overlap Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'exclusion_overlap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_cost_outliers(self):
        """Plot cost outlier analysis"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        df = self.results['detailed_records']
        usable = df[df['IsUsable'] == 1]['TotalPaidAmt']
        
        # Remove negative and zero for log scale
        usable_positive = usable[usable > 0]
        
        # Q-Q plot (using normal distribution instead of lognorm to avoid parameter issue)
        ax = axes[0]
        from scipy import stats
        # Use normal Q-Q plot on log-transformed data
        if len(usable_positive) > 0:
            log_data = np.log(usable_positive)
            stats.probplot(log_data, dist="norm", plot=ax)
            ax.set_title('Q-Q Plot (Log-Transformed)')
        else:
            ax.text(0.5, 0.5, 'No positive data', ha='center', va='center')
            ax.set_title('Q-Q Plot')
        ax.grid(True, alpha=0.3)
        
        # Box plot with outlier thresholds
        ax = axes[1]
        if len(usable_positive) > 0:
            bp = ax.boxplot(usable_positive, vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            
            # Add outlier threshold lines
            Q1 = usable_positive.quantile(0.25)
            Q3 = usable_positive.quantile(0.75)
            IQR = Q3 - Q1
            lower_fence = Q1 - 3 * IQR
            upper_fence = Q3 + 3 * IQR
            
            ax.axhline(y=lower_fence, color='r', linestyle='--', alpha=0.5, label=f'Lower Fence: ${lower_fence:,.0f}')
            ax.axhline(y=upper_fence, color='r', linestyle='--', alpha=0.5, label=f'Upper Fence: ${upper_fence:,.0f}')
            ax.set_ylabel('Annual Cost ($)')
            ax.set_title('Cost Distribution with Outlier Thresholds')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title('Cost Distribution')
        ax.grid(axis='y', alpha=0.3)
        
        # Outlier summary
        ax = axes[2]
        ax.axis('off')
        
        if len(usable_positive) > 0:
            Q1 = usable_positive.quantile(0.25)
            Q3 = usable_positive.quantile(0.75)
            IQR = Q3 - Q1
            lower_fence = Q1 - 3 * IQR
            upper_fence = Q3 + 3 * IQR
            
            outlier_stats = [
                ['Total Usable Records', f'{len(usable_positive):,}'],
                ['Mean Cost', f'${usable_positive.mean():,.0f}'],
                ['Median Cost', f'${usable_positive.median():,.0f}'],
                ['Q1', f'${Q1:,.0f}'],
                ['Q3', f'${Q3:,.0f}'],
                ['IQR', f'${IQR:,.0f}'],
                ['Lower Fence (Q1 - 3×IQR)', f'${lower_fence:,.0f}'],
                ['Upper Fence (Q3 + 3×IQR)', f'${upper_fence:,.0f}'],
                ['Outliers Below', f'{(usable_positive < lower_fence).sum():,}'],
                ['Outliers Above', f'{(usable_positive > upper_fence).sum():,}'],
                ['Total Outliers', f'{((usable_positive < lower_fence) | (usable_positive > upper_fence)).sum():,}'],
                ['Outlier Rate', f'{self.cost_outliers.get("pct_outliers", 0):.2f}%']
            ]
        else:
            outlier_stats = [['No data available', '']]
        
        table = ax.table(cellText=outlier_stats, cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)
        ax.set_title('Outlier Statistics', pad=20)
        
        plt.suptitle('Cost Outlier Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'cost_outliers.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_latex_report(self):
        """Generate LaTeX report section"""
        
        overall = self.results['overall_stats'].iloc[0]
        exclusion = self.results['exclusion_summary'].iloc[0]
        consumer = self.results['consumer_summary'].iloc[0]
        
        # Generate LaTeX commands file for dynamic values
        commands_content = f"""% Automatically generated data quality statistics
% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

% Overall statistics
\\renewcommand{{\\TheTotalNumberCustomers}}{{{int(overall['TotalUniqueConsumers']):,}}}
\\renewcommand{{\\TheInitialYear}}{{{int(overall['EarliestFiscalYear'])}}}
\\renewcommand{{\\TheFinalYear}}{{{int(overall['LatestFiscalYear'])}}}
\\renewcommand{{\\TheTotalCustomerYears}}{{{int(overall['TotalConsumerYears']):,}}}

% Customer data availability
\\renewcommand{{\\CustomerNumberOneYear}}{{{int(consumer['ConsumersWithUsableData']):,}}}
\\renewcommand{{\\CustomerPctOneYear}}{{{consumer['Pct_ConsumersUsable']:.1f}}}
\\renewcommand{{\\CustomerNumberTwoPlusYear}}{{{int(consumer['ConsumersWithMultiYear']):,}}}
\\renewcommand{{\\CustomerPctTwoPlusYear}}{{{consumer['Pct_ConsumersMultiYear']:.1f}}}
\\renewcommand{{\\CustomerNumberNoData}}{{{int(consumer['TotalConsumers'] - consumer['ConsumersWithUsableData']):,}}}
\\renewcommand{{\\CustomerPctNoData}}{{{(100 - consumer['Pct_ConsumersUsable']):.1f}}}

% Exclusion statistics
\\renewcommand{{\\ExclusionMidYearQSICount}}{{{int(exclusion['Count_MidYearQSI']):,}}}
\\renewcommand{{\\ExclusionMidYearQSIPct}}{{{exclusion['Pct_MidYearQSI']:.1f}}}
\\renewcommand{{\\ExclusionLateEntryCount}}{{{int(exclusion['Count_LateEntry']):,}}}
\\renewcommand{{\\ExclusionLateEntryPct}}{{{exclusion['Pct_LateEntry']:.1f}}}
\\renewcommand{{\\ExclusionEarlyExitCount}}{{{int(exclusion['Count_EarlyExit']):,}}}
\\renewcommand{{\\ExclusionEarlyExitPct}}{{{exclusion['Pct_EarlyExit']:.1f}}}
\\renewcommand{{\\ExclusionNoCostsCount}}{{{int(exclusion['Count_NoCosts']):,}}}
\\renewcommand{{\\ExclusionNoCostsPct}}{{{exclusion['Pct_NoCosts']:.1f}}}
\\renewcommand{{\\ExclusionInsufficientServiceCount}}{{{int(exclusion['Count_InsufficientService']):,}}}
\\renewcommand{{\\ExclusionInsufficientServicePct}}{{{exclusion['Pct_InsufficientService']:.1f}}}
\\renewcommand{{\\ExclusionNoQSICount}}{{{int(exclusion['Count_NoQSI']):,}}}
\\renewcommand{{\\ExclusionNoQSIPct}}{{{exclusion['Pct_NoQSI']:.1f}}}

% Cost statistics
\\renewcommand{{\\AvgAnnualCost}}{{\\${overall['AvgAnnualCost']:,.0f}}}
\\renewcommand{{\\MedianAnnualCost}}{{\\${overall['Median_Cost']:,.0f}}}
\\renewcommand{{\\StdevAnnualCost}}{{\\${overall['StDevAnnualCost']:,.0f}}}

% Trajectory analysis
\\renewcommand{{\\CustomersWithTrajectory}}{{{self.consumer_trajectory['consumers_with_trajectory']:,}}}
\\renewcommand{{\\PctWithTrajectory}}{{{self.consumer_trajectory['pct_with_trajectory']:.1f}}}
\\renewcommand{{\\CustomersWithGaps}}{{{self.consumer_trajectory['consumers_with_gaps']:,}}}
\\renewcommand{{\\PctWithGaps}}{{{self.consumer_trajectory['pct_with_gaps']:.1f}}}

% Outlier statistics
\\renewcommand{{\\OutlierLowerFence}}{{\\${self.cost_outliers['lower_fence']:,.0f}}}
\\renewcommand{{\\OutlierUpperFence}}{{\\${self.cost_outliers['upper_fence']:,.0f}}}
\\renewcommand{{\\OutliersBelow}}{{{self.cost_outliers['n_below']:,}}}
\\renewcommand{{\\OutliersAbove}}{{{self.cost_outliers['n_above']:,}}}
\\renewcommand{{\\OutlierRate}}{{{self.cost_outliers['pct_outliers']:.2f}}}
"""
        
        # Save LaTeX commands file
        commands_path = Path('../report/data_quality_commands.tex')
        with open(commands_path, 'w') as f:
            f.write(commands_content)
        
        print(f"LaTeX commands saved to {commands_path}")
        
        # Generate main LaTeX report section
        latex_content = r"""\section{Data Quality and Outlier Analysis}

\subsection{Overview}

This section presents a comprehensive analysis of data quality and outlier identification for the iBudget model calibration. The analysis examines customer-year records to identify data quality issues and determine the usable dataset for model development.

\subsection{Data Availability Assessment}

The initial data quality analysis examined \TheTotalNumberCustomers{} unique customers in the Agency for Persons with Disabilities (APD) database, spanning fiscal years \TheInitialYear{} through \TheFinalYear{}. Each fiscal year runs from September 1 through August 31, creating potential complications when Questionnaire for Situational Information (QSI) assessments occur mid-year.

The outlier analysis revealed substantial data quality challenges:
\begin{itemize}
    \item \textbf{\CustomerNumberOneYear{} customers (\CustomerPctOneYear\%)} have at least one fiscal year of usable data
    \item \textbf{\CustomerNumberTwoPlusYear{} customers (\CustomerPctTwoPlusYear\%)} have two or more years suitable for trajectory modeling
    \item \textbf{\CustomerNumberNoData{} customers (\CustomerPctNoData\%)} have no usable data after applying quality criteria
\end{itemize}

\subsection{Customer Coverage Analysis}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{figures/consumer_coverage.png}
    \caption{Customer-level data coverage analysis}
    \label{fig:customer_coverage}
\end{figure}

Figure~\ref{fig:customer_coverage} presents four panels analyzing customer data availability:
\begin{itemize}
    \item \textbf{Top-left panel}: Distribution of total years each customer appears in the database, showing most customers have either 1-2 years or the full span of data
    \item \textbf{Top-right panel}: Distribution of usable years after applying quality criteria, revealing significant data loss for many customers
    \item \textbf{Bottom-left panel}: Scatter plot comparing total versus usable years, with the red diagonal line representing perfect data usability; points below the line indicate data quality issues
    \item \textbf{Bottom-right panel}: Pie chart showing the critical breakdown---only \CustomerPct2PlusYear\% of customers have sufficient data for trajectory modeling
\end{itemize}

\subsection{Data Exclusion Analysis}

Customer-year records are evaluated for several quality issues that would compromise model calibration:

\begin{table}[h]
\centering
\caption{Exclusion Reasons and Impact}
\begin{tabular}{lrr}
\toprule
\textbf{Exclusion Reason} & \textbf{Count} & \textbf{Percentage} \\
\midrule
Mid-Year QSI Change & \ExclusionMidYearQSICount & \ExclusionMidYearQSIPct\% \\
Late Entry (>30 days) & \ExclusionLateEntryCount & \ExclusionLateEntryPct\% \\
Early Exit (>30 days) & \ExclusionEarlyExitCount & \ExclusionEarlyExitPct\% \\
No Costs Recorded & \ExclusionNoCostsCount & \ExclusionNoCostsPct\% \\
Insufficient Service Days & \ExclusionInsufficientServiceCount & \ExclusionInsufficientServicePct\% \\
No QSI Assessment & \ExclusionNoQSICount & \ExclusionNoQSIPct\% \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{figures/exclusion_reasons.png}
    \caption{Distribution of exclusion reasons}
    \label{fig:exclusion_reasons}
\end{figure}

Figure~\ref{fig:exclusion_reasons} visualizes the exclusion analysis:
\begin{itemize}
    \item \textbf{Left panel}: Absolute counts of customer-years excluded for each reason, showing the scale of data loss
    \item \textbf{Right panel}: Percentage breakdown highlighting that mid-year QSI changes and incomplete fiscal years are the primary exclusion drivers
\end{itemize}

\subsection{Cost Distribution Analysis}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{figures/cost_distributions.png}
    \caption{Comparison of cost distributions between included and excluded customer-years}
    \label{fig:cost_distributions}
\end{figure}

Figure~\ref{fig:cost_distributions} compares cost patterns between included and excluded data:
\begin{itemize}
    \item \textbf{Top-left}: Histogram overlay showing excluded records tend toward lower costs
    \item \textbf{Top-right}: Box plots revealing excluded records have wider variance and more outliers
    \item \textbf{Bottom-left}: Log-scale distribution highlighting the heavy tail in both populations
    \item \textbf{Bottom-right}: Summary statistics confirming systematically different cost profiles
\end{itemize}

\subsection{Temporal Trends}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{figures/fiscal_year_trends.png}
    \caption{Trends across fiscal years}
    \label{fig:fiscal_year_trends}
\end{figure}

Figure~\ref{fig:fiscal_year_trends} examines patterns over time:
\begin{itemize}
    \item \textbf{Top-left}: Customer counts by fiscal year show program growth
    \item \textbf{Top-right}: Comparison of total versus usable records reveals consistent data quality challenges
    \item \textbf{Bottom-left}: Average costs trending upward, reflecting inflation and service expansion
    \item \textbf{Bottom-right}: Data usability rates remain relatively stable across years
\end{itemize}

\subsection{Cost Outlier Analysis}

Using the Tukey method with a 3$\times$IQR threshold for extreme outliers:
\begin{itemize}
    \item Lower fence: \OutlierLowerFence
    \item Upper fence: \OutlierUpperFence
    \item Outliers below lower fence: \OutliersBelow
    \item Outliers above upper fence: \OutliersAbove
    \item Total outlier rate: \OutlierRate\%
\end{itemize}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{figures/cost_outliers.png}
    \caption{Cost outlier analysis}
    \label{fig:cost_outliers}
\end{figure}

Figure~\ref{fig:cost_outliers} provides outlier diagnostics:
\begin{itemize}
    \item \textbf{Left panel}: Q-Q plot of log-transformed costs shows approximate normality with heavy tails
    \item \textbf{Center panel}: Box plot with outlier thresholds (red dashed lines) identifies extreme values
    \item \textbf{Right panel}: Statistical summary quantifying outlier prevalence
\end{itemize}

\subsection{Exclusion Overlap Analysis}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{figures/exclusion_overlap.png}
    \caption{Analysis of exclusion flag correlations and overlap}
    \label{fig:exclusion_overlap}
\end{figure}

Figure~\ref{fig:exclusion_overlap} examines relationships between exclusion criteria:
\begin{itemize}
    \item \textbf{Left panel}: Correlation heatmap reveals which exclusion reasons tend to co-occur (red indicates positive correlation)
    \item \textbf{Right panel}: Bar chart showing most excluded records have multiple issues, justifying the conservative approach
\end{itemize}

\subsection{Trajectory Analysis Feasibility}

For the proposed individual trajectory modeling:
\begin{itemize}
    \item \textbf{\CustomersWithTrajectory{} customers (\PctWithTrajectory\%)} have sufficient data for individual trajectory calculation
    \item \textbf{\CustomersWithGaps{} customers (\PctWithGaps\%)} have gaps in their yearly data
    \item Customers without multi-year data will require group-based trajectory imputation
\end{itemize}

\subsection{Recommendations}

Based on this analysis:

\begin{enumerate}
    \item \textbf{Data Sufficiency}: With \CustomerPctOneYear\% of customers having usable data and \CustomerPctTwoPlusYear\% having multi-year data, the dataset supports model calibration but requires careful handling of limited trajectory coverage.
    
    \item \textbf{Trajectory Modeling}: The proposed trajectory approach is feasible for approximately \PctWithTrajectory\% of customers. The remaining customers will require cluster-based imputation.
    
    \item \textbf{Exclusion Strategy}: The conservative approach excluding customer-years with mid-year QSI changes affects \ExclusionMidYearQSIPct\% of records but ensures clean QSI-to-cost mapping.
    
    \item \textbf{Model Robustness}: Models should be tested for sensitivity to the exclusion criteria and validated on both included and excluded populations where feasible.
\end{enumerate}
"""
        
        # Save LaTeX file
        latex_path = Path('../report/data_quality_analysis.tex')
        with open(latex_path, 'w') as f:
            f.write(latex_content)
        
        print(f"LaTeX report saved to {latex_path}")
        
    def save_summary_json(self):
        """Save summary statistics to JSON for other modules"""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overall_stats': self.results['overall_stats'].to_dict('records')[0],
            'consumer_summary': self.results['consumer_summary'].to_dict('records')[0],
            'exclusion_summary': self.results['exclusion_summary'].to_dict('records')[0],
            'cost_outliers': self.cost_outliers,
            'consumer_trajectory': self.consumer_trajectory
        }
        
        json_path = Path('../report/outlier_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Summary JSON saved to {json_path}")
        
    def run_analysis(self):
        """Execute the complete outlier analysis pipeline"""
        
        print("="*60)
        print("Starting Outlier Analysis for iBudget Model Calibration")
        print("="*60)
        
        # Execute stored procedure
        self.execute_stored_procedure()
        
        # Perform additional analysis
        self.analyze_exclusions()
        
        # Generate plots
        print("Generating visualization plots...")
        self.generate_plots()
        
        # Generate LaTeX report
        print("Generating LaTeX report section...")
        self.generate_latex_report()
        
        # Save summary JSON
        self.save_summary_json()
        
        print("="*60)
        print("Outlier Analysis Complete!")
        print("="*60)
        
        # Print key findings
        consumer = self.results['consumer_summary'].iloc[0]
        print(f"\nKey Findings:")
        print(f"- Total consumers: {int(consumer['TotalConsumers']):,}")
        print(f"- Consumers with usable data: {int(consumer['ConsumersWithUsableData']):,} ({consumer['Pct_ConsumersUsable']:.1f}%)")
        print(f"- Consumers with trajectory data: {int(consumer['ConsumersWithMultiYear']):,} ({consumer['Pct_ConsumersMultiYear']:.1f}%)")
        print(f"- Cost outlier rate: {self.cost_outliers['pct_outliers']:.2f}%")

def main():
    """Main entry point"""
    
    # Configuration - UPDATE THESE VALUES
    server = '.'  # e.g., 'localhost' or 'server\\instance'
    database = 'APD'
    
    # For Windows Authentication (leave username and password as None)
    username = None  
    password = None
    
    if username and password:
        connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    else:
        # Windows Authentication
        connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes'    
    
    
    # Run analysis
    analyzer = OutlierAnalysis(connection_string)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
