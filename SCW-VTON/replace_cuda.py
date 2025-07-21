#!/usr/bin/env python3
# filepath: cuda_to_cpu_converter.py

import os
import re
import glob

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
        try:
            content = file.read()
        except UnicodeDecodeError:
            print(f"Skipping binary file: {filepath}")
            return False
    
    # Pattern 1:  -> device check
    pattern1 = r'\.cuda\(\)'
    if re.search(pattern1, content):
        modified = re.sub(pattern1, '', content)
        
        # Pattern 2:  -> empty
        modified = re.sub(r'\.cuda\([^)]*\)', '', modified)
        
        # Pattern 3: torch.from_numpy(...) -> torch.from_numpy(...)
        modified = re.sub(r'(torch\.from_numpy\([^)]*\))\.cuda\(\)', r'\1', modified)
        
        # Add device variable at the top of the file if not present
        if 'device =' not in modified:
            # Find imports section
            import_section_end = 0
            lines = modified.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('import') or line.startswith('from'):
                    import_section_end = i
            
            # Add device setup after imports
            device_setup = "\nimport torch\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
            lines.insert(import_section_end + 1, device_setup)
            modified = '\n'.join(lines)
        
        # Save the modified content back to the file
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(modified)
        print(f"Modified: {filepath}")
        return True
    
    return False

def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Search for Python files in the SCW-VTON directory
    repo_root = os.path.join(script_dir, "SCW-VTON")
    
    if not os.path.exists(repo_root):
        repo_root = os.path.join(script_dir)
    
    py_files = []
    for root, _, files in os.walk(repo_root):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    
    modified_count = 0
    for py_file in py_files:
        if process_file(py_file):
            modified_count += 1
    
    print(f"\nCompleted! Modified {modified_count} files.")

if __name__ == "__main__":
    main()