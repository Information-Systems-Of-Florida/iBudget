"""
Utility to count lines of code and capture folder structure
Use:  python cloc.py "[a folder]"
"""

import os
import sys

CODE_EXTENSIONS = {'.py', '.tex', '.json'}

def is_code_file(filename):
    return os.path.splitext(filename)[1] in CODE_EXTENSIONS

def count_code_lines(filepath):
    ext = os.path.splitext(filepath)[1]
    count = 0
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if ext == '.py':
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
            count += 1
    return count

def count_lines_in_directory(directory):
    total_lines = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if is_code_file(file):
                filepath = os.path.join(root, file)
                try:
                    lines = count_code_lines(filepath)
                    print(f'{filepath}: {lines}')
                    total_lines += lines
                except Exception as e:
                    print(f'Could not read {filepath}: {e}')
    print(f'Total lines of code: {total_lines}')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python cloc.py <directory>')
        sys.exit(1)
    target_dir = sys.argv[1]
    if not os.path.isdir(target_dir):
        print(f'Error: {target_dir} is not a directory.')
        sys.exit(1)
    count_lines_in_directory(target_dir)
