import os

def rename_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content.replace('BILIO', 'BILIO').replace('bilio', 'bilio')
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated: {filepath}")

def main():
    root_dir = r"c:\Users\Rishith Raswant\OneDrive\Desktop\bilio\bilio-sih26182"
    extensions = {'.py', '.yaml', '.md', '.txt', '.sh', '.json'}
    exclude_dirs = {'.venv', '.git', 'node_modules', '__pycache__'}
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Exclude directories
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in extensions:
                filepath = os.path.join(dirpath, filename)
                try:
                    rename_in_file(filepath)
                except Exception as e:
                    print(f"Error on {filepath}: {e}")

if __name__ == '__main__':
    main()
