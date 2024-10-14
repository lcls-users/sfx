import argparse
import os

def generate_context(project_descriptor, exclude_patterns):
    with open('context', 'w') as output_file:
        for root, _, files in os.walk('.'):
            for file in files:
                if file.endswith('.py') and not any(exclude in file for exclude in exclude_patterns):
                    file_path = os.path.join(root, file)
                    output_file.write(f'<file path="{file_path}" project="{project_descriptor}">\n')
                    with open(file_path, 'r') as f:
                        output_file.write(f.read())
                    output_file.write('</file>\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate project context while excluding specific files.")
    parser.add_argument('project_descriptor', help='Project descriptor name')
    parser.add_argument('--exclude', nargs='+', default=[], help='List of file patterns to exclude')

    args = parser.parse_args()

    generate_context(args.project_descriptor, args.exclude)

