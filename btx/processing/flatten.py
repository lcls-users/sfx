import argparse
import os

def generate_context(project_descriptor, exclude_patterns=None, include_files=None):
    with open('context', 'w') as output_file:
        if include_files:
            # Mode for explicitly named files
            for file_name in include_files:
                for root, _, files in os.walk('.'):
                    if file_name in files:
                        file_path = os.path.join(root, file_name)
                        write_file_content(output_file, file_path, project_descriptor)
                        break  # Stop searching once the file is found
        else:
            # Original mode: walk through all files
            for root, _, files in os.walk('.'):
                for file in files:
                    if file.endswith('.py') and (not exclude_patterns or not any(exclude in file for exclude in exclude_patterns)):
                        file_path = os.path.join(root, file)
                        write_file_content(output_file, file_path, project_descriptor)

def write_file_content(output_file, file_path, project_descriptor):
    output_file.write(f'<file path="{file_path}" project="{project_descriptor}">\n')
    with open(file_path, 'r') as f:
        output_file.write(f.read())
    output_file.write('</file>\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate project context with options to exclude or include specific files.")
    parser.add_argument('project_descriptor', help='Project descriptor name')
    parser.add_argument('--exclude', nargs='+', default=[], help='List of file patterns to exclude')
    parser.add_argument('--include', nargs='+', help='List of specific file names to include (overrides exclude)')

    args = parser.parse_args()

    generate_context(args.project_descriptor, args.exclude, args.include)

