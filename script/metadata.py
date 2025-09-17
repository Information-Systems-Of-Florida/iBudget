import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import csv
from sqlalchemy import create_engine
import urllib

class DatabaseMetadataExtractor:
    def __init__(self, server, database, username=None, password=None):
        """
        Initialize connection to SQL Server database.
        If username/password are not provided, uses Windows Authentication.
        """
        if username and password:
            self.conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
        else:
            # Windows Authentication
            self.conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes'
        
        # Create SQLAlchemy engine for pandas compatibility
        params = urllib.parse.quote_plus(self.conn_str)
        self.engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
        
        self.database = database
        self.conn = None
        self.data_dictionary = {}
        self.numeric_columns = []  # Store info about numeric columns for histograms
        
    def load_data_dictionary(self, dict_path='../assets/DataDictionary.txt'):
        """Load the data dictionary from the text file"""
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    table_name = row['Table Name'].strip()
                    column_name = row['Data Element Name'].strip()
                    description = row['Description'].strip(',').strip()
                    
                    if table_name not in self.data_dictionary:
                        self.data_dictionary[table_name] = {}
                    
                    self.data_dictionary[table_name][column_name] = description
            
            print(f"Loaded data dictionary with {len(self.data_dictionary)} tables")
            return True
        except Exception as e:
            print(f"Error loading data dictionary: {e}")
            print("Continuing without descriptions...")
            return False
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = pyodbc.connect(self.conn_str)
            print(f"Successfully connected to {self.database}")
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False
    
    def get_tables(self):
        """Get all tables in the database"""
        query = """
        SELECT TABLE_SCHEMA, TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_TYPE = 'BASE TABLE' 
        ORDER BY TABLE_SCHEMA, TABLE_NAME
        """
        cursor = self.conn.cursor()
        cursor.execute(query)
        tables = cursor.fetchall()
        return [(row[0], row[1]) for row in tables]
    
    def get_table_info(self, schema, table_name):
        """Get basic table information"""
        # Get row count
        count_query = f"SELECT COUNT(*) FROM [{schema}].[{table_name}]"
        cursor = self.conn.cursor()
        cursor.execute(count_query)
        row_count = cursor.fetchone()[0]
        
        # Get column count
        col_query = """
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        """
        cursor.execute(col_query, (schema, table_name))
        col_count = cursor.fetchone()[0]
        
        return row_count, col_count
    
    def get_column_description(self, table_name, column_name):
        """Get column description from data dictionary"""
        if table_name in self.data_dictionary:
            return self.data_dictionary[table_name].get(column_name, '')
        return ''
    
    def get_unique_count(self, schema, table_name, column_name):
        """Get count of unique values in a column"""
        try:
            query = f"""
            SELECT COUNT(DISTINCT [{column_name}]) 
            FROM [{schema}].[{table_name}]
            """
            cursor = self.conn.cursor()
            cursor.execute(query)
            return cursor.fetchone()[0]
        except:
            return 0
    
    def get_column_metadata(self, schema, table_name):
        """Get detailed column metadata"""
        # Get column information
        col_query = """
        SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH,
               NUMERIC_PRECISION, NUMERIC_SCALE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        ORDER BY ORDINAL_POSITION
        """
        cursor = self.conn.cursor()
        cursor.execute(col_query, (schema, table_name))
        columns = cursor.fetchall()
        
        metadata = []
        
        for col in columns:
            col_name = col[0]
            data_type = col[1]
            
            # Format data type string
            if col[2]:  # Character types with length
                data_type_str = f"{data_type}({col[2]})"
            elif col[3]:  # Numeric types with precision
                if col[4]:
                    data_type_str = f"{data_type}({col[3]},{col[4]})"
                else:
                    data_type_str = f"{data_type}({col[3]})"
            else:
                data_type_str = data_type
            
            # Get description from data dictionary
            description = self.get_column_description(table_name, col_name)
            
            # Get unique count
            unique_count = self.get_unique_count(schema, table_name, col_name)
            
            # Determine if numeric or date
            numeric_types = ['int', 'bigint', 'smallint', 'tinyint', 'decimal', 
                           'numeric', 'float', 'real', 'money', 'smallmoney', 'bit']
            date_types = ['date', 'datetime', 'datetime2', 'time', 'timestamp']
            
            is_numeric = any(nt in data_type.lower() for nt in numeric_types)
            is_date = any(dt in data_type.lower() for dt in date_types)
            
            # Get column statistics
            stats = self.get_column_stats(schema, table_name, col_name, is_numeric, is_date)
            
            # Store numeric column info for histogram generation
            if is_numeric and stats != "All NULL values" and stats != "No numeric data":
                self.numeric_columns.append({
                    'schema': schema,
                    'table': table_name,
                    'column': col_name,
                    'description': description
                })
            
            metadata.append({
                'column_name': col_name,
                'data_type': data_type_str,
                'description': description,
                'unique_count': unique_count,
                'stats': stats
            })
        
        return metadata
    
    def get_column_stats(self, schema, table_name, col_name, is_numeric, is_date):
        """Get statistics for a specific column"""
        cursor = self.conn.cursor()
        
        # Check for NULL values first
        null_query = f"SELECT COUNT(*) FROM [{schema}].[{table_name}] WHERE [{col_name}] IS NULL"
        cursor.execute(null_query)
        null_count = cursor.fetchone()[0]
        
        # Get total count
        total_query = f"SELECT COUNT(*) FROM [{schema}].[{table_name}]"
        cursor.execute(total_query)
        total_count = cursor.fetchone()[0]
        
        if null_count == total_count:
            return "All NULL values"
        
        if is_date:
            # Get date range
            date_query = f"""
            SELECT 
                MIN([{col_name}]) as min_date,
                MAX([{col_name}]) as max_date
            FROM [{schema}].[{table_name}]
            WHERE [{col_name}] IS NOT NULL
            """
            try:
                cursor.execute(date_query)
                result = cursor.fetchone()
                if result and result[0] is not None:
                    min_date = result[0].strftime('%Y-%m-%d') if hasattr(result[0], 'strftime') else str(result[0])
                    max_date = result[1].strftime('%Y-%m-%d') if hasattr(result[1], 'strftime') else str(result[1])
                    return f"Range: [{min_date}, {max_date}]"
                else:
                    return "No date data"
            except:
                return "Error computing date range"
        
        elif is_numeric:
            # Get numeric statistics
            stats_query = f"""
            SELECT 
                MIN(CAST([{col_name}] AS FLOAT)) as min_val,
                MAX(CAST([{col_name}] AS FLOAT)) as max_val,
                AVG(CAST([{col_name}] AS FLOAT)) as avg_val
            FROM [{schema}].[{table_name}]
            WHERE [{col_name}] IS NOT NULL
            """
            
            try:
                cursor.execute(stats_query)
                result = cursor.fetchone()
                
                if result and result[0] is not None:
                    min_val = result[0]
                    max_val = result[1]
                    avg_val = result[2]
                    
                    # Get median
                    median_query = f"""
                    SELECT CAST([{col_name}] AS FLOAT) as val
                    FROM [{schema}].[{table_name}]
                    WHERE [{col_name}] IS NOT NULL
                    ORDER BY [{col_name}]
                    OFFSET (SELECT COUNT(*)/2 FROM [{schema}].[{table_name}] WHERE [{col_name}] IS NOT NULL) ROWS
                    FETCH NEXT 1 ROWS ONLY
                    """
                    
                    try:
                        cursor.execute(median_query)
                        median_result = cursor.fetchone()
                        median_val = median_result[0] if median_result else avg_val
                    except:
                        median_val = avg_val
                    
                    return f"Range: [{min_val:.2f}, {max_val:.2f}], Avg: {avg_val:.2f}, Median: {median_val:.2f}"
                else:
                    return "No numeric data"
            except:
                return "Error computing statistics"
        else:
            # Get unique values for non-numeric columns
            unique_query = f"""
            SELECT DISTINCT TOP 11 [{col_name}]
            FROM [{schema}].[{table_name}]
            WHERE [{col_name}] IS NOT NULL
            """
            
            cursor.execute(unique_query)
            unique_values = cursor.fetchall()
            
            if len(unique_values) > 10:
                return "..."
            elif len(unique_values) == 0:
                return "No non-NULL values"
            else:
                # Format unique values
                values = [str(row[0]) for row in unique_values]
                # Escape LaTeX special characters
                values = [self.escape_latex(v[:50]) for v in values]  # Limit length
                return f"\\{{{', '.join(values)}\\}}"
    
    def escape_latex(self, text):
        """Escape special LaTeX characters"""
        special_chars = ['\\', '#', '$', '%', '&', '_', '{', '}', '^', '~']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text
    
    def generate_histogram(self, schema, table_name, column_name, description, output_dir='../report/figures'):
        """Generate histogram for a numeric column"""
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            # Fetch the data using SQLAlchemy engine for pandas
            query = f"""
            SELECT CAST([{column_name}] AS FLOAT) as val
            FROM [{schema}].[{table_name}]
            WHERE [{column_name}] IS NOT NULL
            """
            
            df = pd.read_sql_query(query, self.engine)
            
            if len(df) > 0:
                # Create histogram
                plt.figure(figsize=(8, 6))
                plt.hist(df['val'], bins=30, edgecolor='black', alpha=0.7)
                
                # Add title and labels
                title = f"{table_name}.{column_name}"
                if description:
                    title += f"\n{description}"
                
                plt.title(title, fontsize=12)
                plt.xlabel(column_name, fontsize=10)
                plt.ylabel('Frequency', fontsize=10)
                plt.grid(True, alpha=0.3)
                
                # Save the figure
                filename = f"{schema}_{table_name}_{column_name}.pdf"
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, bbox_inches='tight')
                plt.close()
                
                return filename
        except Exception as e:
            print(f"Error generating histogram for {table_name}.{column_name}: {e}")
            return None
        
        return None
    
    def generate_latex(self, output_file='../report/APD_metadata.tex'):
        """Generate LaTeX document with database metadata"""
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load data dictionary first
        self.load_data_dictionary()
        
        if not self.connect():
            return False
        
        tables = self.get_tables()
        
        # Start LaTeX content (only sections, no document structure)
        latex_content = []
        latex_content.append("\\section{Data Provided by ADP for Modeling}")
        latex_content.append("\\begin{landscape}")
        latex_content.append("")
        
        # Process each table
        for schema, table_name in tables:
            print(f"Processing table: {schema}.{table_name}")
            
            # Get table information
            row_count, col_count = self.get_table_info(schema, table_name)
            
            # Add subsection for this table
            latex_content.append(f"\\subsection{{{self.escape_latex(f'{table_name}')}}}")
            latex_content.append("")
            latex_content.append("\\subsubsection{Table Overview}")
            latex_content.append("\\begin{itemize}")
            latex_content.append(f"\\item \\textbf{{Table Name:}} {self.escape_latex(table_name)}")
            latex_content.append(f"\\item \\textbf{{Schema:}} {self.escape_latex(schema)}")
            latex_content.append(f"\\item \\textbf{{Number of Records:}} {row_count:,}")
            latex_content.append(f"\\item \\textbf{{Number of Columns:}} {col_count}")
            latex_content.append("\\end{itemize}")
            latex_content.append("")
            
            # Get column metadata
            columns = self.get_column_metadata(schema, table_name)
            
            # Create columns table with Description and N columns
            latex_content.append("\\subsubsection{Column Details}")
            latex_content.append("\\begin{longtable}{|l|l|l|r|p{6cm}|}")
            latex_content.append("\\hline")
            latex_content.append("\\textbf{Column Name} & \\textbf{Data Type} & \\textbf{Description} & \\textbf{N} & \\textbf{Statistics/Values} \\\\")
            latex_content.append("\\hline")
            latex_content.append("\\endfirsthead")
            latex_content.append("\\multicolumn{5}{c}{\\textit{Continued from previous page}} \\\\")
            latex_content.append("\\hline")
            latex_content.append("\\textbf{Column Name} & \\textbf{Data Type} & \\textbf{Description} & \\textbf{N} & \\textbf{Statistics/Values} \\\\")
            latex_content.append("\\hline")
            latex_content.append("\\endhead")
            latex_content.append("\\hline")
            latex_content.append("\\multicolumn{5}{c}{\\textit{Continued on next page}} \\\\")
            latex_content.append("\\endfoot")
            latex_content.append("\\hline")
            latex_content.append("\\endlastfoot")
            
            for col in columns:
                col_name = self.escape_latex(col['column_name'])
                data_type = self.escape_latex(col['data_type'])
                description = self.escape_latex(col['description'])[:100]  # Limit description length
                unique_count = col['unique_count']
                stats = col['stats'] if '\\{' in col['stats'] else self.escape_latex(col['stats'])
                
                latex_content.append(f"{col_name} & {data_type} & {description} & {unique_count} & {stats} \\\\")
                latex_content.append("\\hline")
            
            latex_content.append("\\end{longtable}")
            latex_content.append("")
        latex_content.append("\\end{landscape}")
        
        # Generate histograms section
        if self.numeric_columns:
            latex_content.append("")
            latex_content.append("\\newpage \\section{Histograms}")
            latex_content.append("")
            
            print("\nGenerating histograms...")
            for col_info in self.numeric_columns:
                schema = col_info['schema']
                table = col_info['table']
                column = col_info['column']
                description = col_info['description']
                
                print(f"  Generating histogram for {table}.{column}")
                
                # Generate histogram
                filename = self.generate_histogram(schema, table, column, description)
                
                if filename:
                    # Add to LaTeX
                    latex_content.append(f"\\subsection{{{self.escape_latex(f'{table}.{column}')}}}")
                    if description:
                        latex_content.append(f"\\textit{{{self.escape_latex(description)}}}")
                    latex_content.append("")
                    latex_content.append("\\begin{figure}[htbp]")
                    latex_content.append("\\centering")
                    latex_content.append(f"\\includegraphics[width=\\textwidth]{{figures/{filename}}}")
                    latex_content.append(f"\\caption{{{self.escape_latex(f'Distribution of {column} in {table}')}}}")
                    latex_content.append("\\end{figure}\\newpage")
                    latex_content.append("")
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(latex_content))
        
        print(f"\nLaTeX content generated: {output_file}")
        print(f"Histograms saved in: ../report/figures/")
        print("\nNote: This file contains only sections. Include it in your main LaTeX document using:")
        print(f"  \\input{{APD_metadata.tex}}")  # Assuming main doc is in report folder
        
        self.conn.close()
        return True

# Main execution
if __name__ == "__main__":
    # Configuration - UPDATE THESE VALUES
    SERVER = '.'  # e.g., 'localhost' or 'server\\instance'
    DATABASE = 'APD'
    
    # For Windows Authentication (leave username and password as None)
    USERNAME = None  
    PASSWORD = None
    
    # For SQL Server Authentication (uncomment and fill in)
    # USERNAME = 'your_username'
    # PASSWORD = 'your_password'
    
    # Note: Install required packages with:
    # pip install pyodbc pandas numpy matplotlib sqlalchemy
    
    # Create extractor and generate LaTeX
    extractor = DatabaseMetadataExtractor(SERVER, DATABASE, USERNAME, PASSWORD)
    
    # Generate the LaTeX document (saves to ../report/ folder)
    extractor.generate_latex()  # Uses default path ../report/APD_metadata.tex
    
    print("\nDone! The LaTeX sections have been created in ../report/")
    print("\nTo use in your main document:")
    print("1. Ensure you have the following packages in your preamble:")
    print("   \\usepackage{longtable}")
    print("   \\usepackage{graphicx}")
    print("2. Include the generated content with: \\input{APD_metadata.tex} (from within the report folder)")