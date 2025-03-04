import os

# Define the folder structure
folders = [
    'data',
    'notebooks',
    'experiments/logs',
    'experiments/models',
    'src',
    'docs/presentations',
    'docs/references'
]

# Create the folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create basic files
basic_files = ['.gitignore', 'requirements.txt', 'README.md']

for file in basic_files:
    with open(file, 'w') as f:
        if file == 'README.md':
            f.write("# Side-Channel Analysis Project\n\n## Overview\n\nThis project focuses on side-channel analysis using deep learning techniques applied to datasets like ASCAD.\n")
        elif file == '.gitignore':
            f.write("__pycache__/\n*.log\n*.zip\n*.h5\nvenv/\n")
        elif file == 'requirements.txt':
            f.write("# Add your project dependencies here\n")

print("New project structure created successfully!")