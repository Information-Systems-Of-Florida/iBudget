import os

def sanitize_file(input_path, output_path):
    with open(input_path, 'rb') as infile:
        raw = infile.read()
    try:
        text = raw.decode('utf-8', errors='replace')
    except UnicodeDecodeError:
        text = raw.decode('latin1', errors='replace')
    text = text.replace('²', '^2')
    text = text.replace('±', '+-')
    ascii_text = text.encode('ascii', errors='replace').decode('ascii')
    # Split and rejoin lines to normalize
    lines = ascii_text.splitlines()
    with open(output_path, 'w', encoding='ascii', newline='\n') as outfile:
        for line in lines:
            outfile.write(line + '\n')

log_dir = '../report/logs'
source_dir = '../script/models'

# Clean logs
for filename in os.listdir(log_dir):
    input_path = os.path.join(log_dir, filename)
    if not os.path.isfile(input_path):
        continue
    name, ext = os.path.splitext(filename)
    if name.endswith('_ascii'):
        output_filename = filename  # Do not add suffix again
    else:
        output_filename = f'{name}_ascii{ext}'
    output_path = os.path.join(log_dir, output_filename)
    sanitize_file(input_path, output_path)

# Clean source files, copy to logs
for filename in os.listdir(source_dir):
    if not filename.endswith('.py'):
        continue
    input_path = os.path.join(source_dir, filename)
    if not os.path.isfile(input_path):
        continue
    name, ext = os.path.splitext(filename)
    if name.endswith('_ascii'):
        output_filename = filename  # Do not add suffix again
    else:
        output_filename = f'{name}_ascii{ext}'
    output_path = os.path.join(log_dir, output_filename)
    sanitize_file(input_path, output_path)
