#!/bin/bash

echo "================================================================"
echo "Applying Fixed Minimal Unsloth to Python Files"
echo "================================================================"

# Directory paths
SRC_DIR="/notebooks/src/generative_ai_module"
CUSTOM_UNSLOTH_DIR="/notebooks/custom_unsloth"

# Make sure custom unsloth directory exists
if [ ! -d "$CUSTOM_UNSLOTH_DIR" ]; then
    echo "❌ Error: Custom unsloth directory not found at $CUSTOM_UNSLOTH_DIR"
    echo "Creating custom unsloth directory now..."
    
    # Run the create_minimal_unsloth.sh script
    ./setup/create_minimal_unsloth.sh
    
    if [ ! -d "$CUSTOM_UNSLOTH_DIR" ]; then
        echo "❌ Error: Failed to create custom unsloth directory"
        exit 1
    fi
fi

# Make sure the src directory exists
if [ ! -d "$SRC_DIR" ]; then
    echo "⚠️ Warning: Source directory not found at $SRC_DIR"
    echo "Please enter the path to your code directory:"
    read -p "> " SRC_DIR
    
    if [ ! -d "$SRC_DIR" ]; then
        echo "❌ Error: Directory not found at $SRC_DIR"
        exit 1
    fi
fi

# Step 1: First fix any indentation errors in jarvis_unified.py
echo "Step 1: Fixing indentation errors in jarvis_unified.py..."
python setup/fix_jarvis_unified.py "$SRC_DIR/jarvis_unified.py"

# Step 2: Adjust Python files to use minimal unsloth
echo "Step 2: Adjusting Python files to use minimal unsloth..."

# First create a list of files that definitely use unsloth
echo "Finding files that use unsloth..."
UNSLOTH_FILES=$(grep -l -r "unsloth" --include="*.py" "$SRC_DIR" || echo "")

if [ -z "$UNSLOTH_FILES" ]; then
    echo "⚠️ Warning: No files found that use unsloth. Trying to scan all Python files."
    find "$SRC_DIR" -name "*.py" > /tmp/python_files.txt
else
    echo "$UNSLOTH_FILES" > /tmp/python_files.txt
    echo "Found $(wc -l < /tmp/python_files.txt) files that use unsloth"
fi

# Process each file individually
while read -r file; do
    echo "Processing $file..."
    
    # Create backup if it doesn't exist
    if [ ! -f "${file}.bak" ]; then
        cp "$file" "${file}.bak"
        echo "Created backup: ${file}.bak"
    fi
    
    # Fix the file by adding the necessary imports at the top
    cat > /tmp/temp_header.py << EOF
import sys
import os
# Use custom minimal unsloth implementation
if "$CUSTOM_UNSLOTH_DIR" not in sys.path:
    sys.path.insert(0, "$CUSTOM_UNSLOTH_DIR")

EOF

    # Check if the file already has our custom path
    if ! grep -q "$CUSTOM_UNSLOTH_DIR" "$file"; then
        # Add it to the top of the file (after any possible shebang or docstring)
        if head -1 "$file" | grep -q "^#!"; then
            # File has shebang, insert after it
            sed -i.tmp '1r /tmp/temp_header.py' "$file"
        elif head -10 "$file" | grep -q '"""'; then
            # File has docstring, need to insert after it
            # This is more complex, use a Python script
            python -c "
import re
with open('$file', 'r') as f:
    content = f.read()
if content.startswith('\"\"\"'):
    match = re.search(r'^(\"\"\".*?\"\"\")(.*)$', content, re.DOTALL)
    if match:
        with open('$file', 'w') as f:
            docstring, rest = match.groups()
            with open('/tmp/temp_header.py', 'r') as header:
                header_content = header.read()
            f.write(docstring + '\n' + header_content + rest)
else:
    with open('$file', 'r') as f:
        content = f.read()
    with open('$file', 'w') as f:
        with open('/tmp/temp_header.py', 'r') as header:
            header_content = header.read()
        f.write(header_content + content)
"
        else
            # No shebang or docstring, insert at the top
            cat /tmp/temp_header.py "$file" > "${file}.new"
            mv "${file}.new" "$file"
        fi
    fi
    
    # Add the comment before any unsloth import
    python -c "
import re
with open('$file', 'r') as f:
    content = f.read()
    
# Find all unsloth imports, preserving indentation
unsloth_lines = re.finditer(r'(^|\n)(\s*)(from\s+unsloth\s+import|import\s+unsloth\b)', content, re.MULTILINE)
positions = []
for match in unsloth_lines:
    start_pos = match.start(3)
    indentation = match.group(2)
    positions.append((start_pos, indentation))

# Modify the content from the end to avoid position shifts
for start_pos, indentation in sorted(positions, reverse=True):
    comment = '# Using minimal unsloth implementation\n' + indentation
    content = content[:start_pos] + comment + content[start_pos:]

with open('$file', 'w') as f:
    f.write(content)
"
    echo "✅ Modified $file to use minimal unsloth"
done < /tmp/python_files.txt

# Clean up
rm -f /tmp/temp_header.py /tmp/python_files.txt

echo "================================================================"
echo "Setup Complete!"
echo ""
echo "Remember to activate the minimal unsloth environment before running your code:"
echo "source $CUSTOM_UNSLOTH_DIR/activate_minimal_unsloth.sh"
echo ""
echo "If you need to restore the original Python files, you can use the .bak backups"
echo "================================================================" 