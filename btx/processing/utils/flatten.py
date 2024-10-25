import argparse
import os
from pathlib import Path

def is_path_match(path, pattern):
    """
    Check if path matches pattern, handling both files and directories.
    Ensures proper directory boundary matching.
    """
    path = str(Path(path).resolve())
    pattern = str(Path(pattern).resolve())
    
    # Exact match
    if path == pattern:
        return True
        
    # Check if pattern is a parent directory of path
    # Add trailing slash to ensure directory boundary
    if pattern.endswith(os.sep):
        return path.startswith(pattern)
    return path.startswith(pattern + os.sep)

def should_process_path(path, include_patterns, exclude_patterns):
    """
    Determine if a path should be processed based on include/exclude patterns.
    Returns True if the path should be processed, False otherwise.
    """
    path = Path(path).resolve()
    
    # If no patterns specified, process everything
    if not include_patterns and not exclude_patterns:
        return True
        
    # Check exclusions first
    if exclude_patterns:
        for excl in exclude_patterns:
            if is_path_match(path, excl):
                return False
    
    # If include patterns specified, path must match one
    if include_patterns:
        return any(is_path_match(path, incl) for incl in include_patterns)
    
    return True

def generate_context(project_descriptor, exclude_patterns=None, include_patterns=None):
    """
    Generate context file based on include/exclude patterns.
    """
    base_path = Path('.').resolve()
    
    # Convert patterns to absolute paths
    exclude_patterns = [Path(p).resolve() for p in (exclude_patterns or [])]
    include_patterns = [Path(p).resolve() for p in (include_patterns or [])]
    
    # Debug output
    print("Base path:", base_path)
    print("Exclude patterns:", exclude_patterns)
    print("Include patterns:", include_patterns)
    
    with open('context', 'w') as output_file:
        for root, _, files in os.walk('.'):
            root_path = Path(root).resolve()
            
            for file in files:
                if not file.endswith('.py'):
                    continue
                    
                file_path = root_path / file
                
                # Debug output
                print(f"Checking file: {file_path}")
                print(f"Should process: {should_process_path(file_path, include_patterns, exclude_patterns)}")
                
                if should_process_path(file_path, include_patterns, exclude_patterns):
                    rel_path = file_path.relative_to(base_path)
                    write_file_content(output_file, rel_path, project_descriptor)

def write_file_content(output_file, file_path, project_descriptor):
    """Write file content with proper path handling."""
    output_file.write(f'<file path="{file_path}" project="{project_descriptor}">\n')
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            output_file.write(f.read())
    except UnicodeDecodeError:
        print(f"Warning: Unable to read {file_path} as UTF-8")
    output_file.write('</file>\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate project context with improved path handling."
    )
    parser.add_argument('project_descriptor', help='Project descriptor name')
    parser.add_argument('--exclude', nargs='+', default=[], 
                       help='Files or directories to exclude')
    parser.add_argument('--include', nargs='+', default=[],
                       help='Files or directories to explicitly include')
    
    args = parser.parse_args()
    
    generate_context(args.project_descriptor, args.exclude, args.include)

