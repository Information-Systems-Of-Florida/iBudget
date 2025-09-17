import pyodbc
import pandas as pd
import numpy as np
from datetime import datetime
import os

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
        
        self.database = database
        self.conn = None
        
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
            
            # Determine if numeric
            numeric_types = ['int', 'bigint', 'smallint', 'tinyint', 'decimal', 
                           'numeric', 'float', 'real', 'money', 'smallmoney', 'bit']
            is_numeric = any(nt in data_type.lower() for nt in numeric_types)
            
            # Get column statistics
            stats = self.get_column_stats(schema, table_name, col_name, is_numeric)
            
            metadata.append({
                'column_name': col_name,
                'data_type': data_type_str,
                'stats': stats
            })
        
        return metadata
    
    def get_column_stats(self, schema, table_name, col_name, is_numeric):
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
        
        if is_numeric:
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
                    
                    # Get median (simplified - for exact median you might need different approach)
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
                return "More than 10 unique values"
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
    
    def generate_latex(self, output_file='metadata_output.tex'):
        """Generate LaTeX document with database metadata"""
        if not self.connect():
            return False
        
        tables = self.get_tables()
        
        # Start LaTeX document
        latex_content = []
        # latex_content.append("\\documentclass[11pt]{article}")
        # latex_content.append("\\usepackage[margin=1in]{geometry}")
        # latex_content.append("\\usepackage{longtable}")
        # latex_content.append("\\usepackage{array}")
        # latex_content.append("\\usepackage{booktabs}")
        # latex_content.append("\\usepackage[utf8]{inputenc}")
        # latex_content.append("\\usepackage{hyperref}")
        # latex_content.append("")
        # latex_content.append("\\title{Database Metadata Report: " + self.escape_latex(self.database) + "}")
        # latex_content.append("\\date{\\today}")
        # latex_content.append("")
        # latex_content.append("\\begin{document}")
        # latex_content.append("\\maketitle")
        # latex_content.append("\\tableofcontents")
        # latex_content.append("\\newpage")
        latex_content.append("")
        
        # Process each table
        for schema, table_name in tables:
            print(f"Processing table: {schema}.{table_name}")
            
            # Get table information
            row_count, col_count = self.get_table_info(schema, table_name)
            
            # Add section for this table
            latex_content.append(f"\\section{{{self.escape_latex(f'{schema}.{table_name}')}}}")
            latex_content.append("")
            latex_content.append("\\subsection{Table Overview}")
            latex_content.append("\\begin{itemize}")
            latex_content.append(f"\\item \\textbf{{Table Name:}} {self.escape_latex(table_name)}")
            latex_content.append(f"\\item \\textbf{{Schema:}} {self.escape_latex(schema)}")
            latex_content.append(f"\\item \\textbf{{Number of Records:}} {row_count:,}")
            latex_content.append(f"\\item \\textbf{{Number of Columns:}} {col_count}")
            latex_content.append("\\end{itemize}")
            latex_content.append("")
            
            # Get column metadata
            columns = self.get_column_metadata(schema, table_name)
            
            # Create columns table
            latex_content.append("\\subsection{Column Details}")
            latex_content.append("\\begin{longtable}{|l|l|p{8cm}|}")
            latex_content.append("\\hline")
            latex_content.append("\\textbf{Column Name} & \\textbf{Data Type} & \\textbf{Statistics/Values} \\\\")
            latex_content.append("\\hline")
            latex_content.append("\\endfirsthead")
            latex_content.append("\\multicolumn{3}{c}{\\textit{Continued from previous page}} \\\\")
            latex_content.append("\\hline")
            latex_content.append("\\textbf{Column Name} & \\textbf{Data Type} & \\textbf{Statistics/Values} \\\\")
            latex_content.append("\\hline")
            latex_content.append("\\endhead")
            latex_content.append("\\hline")
            latex_content.append("\\multicolumn{3}{c}{\\textit{Continued on next page}} \\\\")
            latex_content.append("\\endfoot")
            latex_content.append("\\hline")
            latex_content.append("\\endlastfoot")
            
            for col in columns:
                col_name = self.escape_latex(col['column_name'])
                data_type = self.escape_latex(col['data_type'])
                stats = col['stats'] if '\\{' in col['stats'] else self.escape_latex(col['stats'])
                
                latex_content.append(f"{col_name} & {data_type} & {stats} \\\\")
                latex_content.append("\\hline")
            
            latex_content.append("\\end{longtable}")
            latex_content.append("\\newpage")
            latex_content.append("")
        
        # End document
        # latex_content.append("\\end{document}")
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(latex_content))
        
        print(f"\nLaTeX document generated: {output_file}")
        print(f"To compile: pdflatex {output_file}")
        
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
    
    # Create extractor and generate LaTeX
    extractor = DatabaseMetadataExtractor(SERVER, DATABASE, USERNAME, PASSWORD)
    
    # Generate the LaTeX document
    extractor.generate_latex('APD_metadata.tex')
    
    print("\nDone! The LaTeX document has been created.")
    print("You can compile it using: pdflatex APD_metadata.tex")