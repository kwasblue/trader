"""
Enhanced Project Structure Scanner
Scans Python codebase and outputs structure to console and/or file
"""

import ast
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse

try:
    import pathspec
    HAS_PATHSPEC = True
except ImportError:
    HAS_PATHSPEC = False
    print("⚠️  pathspec not installed. Install with: pip install pathspec")


# ============================================================================
# GITIGNORE HANDLING
# ============================================================================

def load_gitignore(project_root: Path) -> Optional['pathspec.PathSpec']:
    """Load .gitignore patterns if pathspec is available"""
    if not HAS_PATHSPEC:
        return None
    
    gitignore_path = project_root / ".gitignore"
    if not gitignore_path.exists():
        return None
    
    try:
        with open(gitignore_path, "r", encoding="utf-8") as f:
            return pathspec.PathSpec.from_lines("gitwildmatch", f)
    except Exception as e:
        print(f"Warning: Failed to load .gitignore: {e}")
        return None


# ============================================================================
# AST PARSING
# ============================================================================

def attach_parents(tree: ast.AST) -> None:
    """Attach parent references to all AST nodes"""
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            setattr(child, "parent", node)


def extract_defs_from_file(filepath: Path) -> List[Dict]:
    """
    Extract classes and functions from a Python file.
    
    Returns:
        List of dicts with 'type', 'name', and optional 'methods'
    """
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()
        
        tree = ast.parse(content, filename=str(filepath))
    except SyntaxError as e:
        print(f"❌ SyntaxError in {filepath}: {e}")
        return []
    except Exception as e:
        print(f"❌ Error parsing {filepath}: {e}")
        return []

    attach_parents(tree)

    results = []
    
    # Walk the tree
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Extract methods
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    # Get function signature
                    args = [arg.arg for arg in item.args.args]
                    methods.append({
                        "name": item.name,
                        "args": args,
                        "lineno": item.lineno
                    })
            
            results.append({
                "type": "class",
                "name": node.name,
                "methods": methods,
                "lineno": node.lineno
            })
        
        elif isinstance(node, ast.FunctionDef):
            # Only top-level functions (not methods)
            parent = getattr(node, "parent", None)
            if not isinstance(parent, ast.ClassDef):
                args = [arg.arg for arg in node.args.args]
                results.append({
                    "type": "function",
                    "name": node.name,
                    "args": args,
                    "lineno": node.lineno
                })
    
    # Sort by line number
    results.sort(key=lambda x: x["lineno"])
    
    return results


# ============================================================================
# DIRECTORY SCANNING
# ============================================================================

def should_ignore_path(rel_path: str) -> bool:
    """Check if path should be ignored (common patterns)"""
    ignore_patterns = [
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        ".env",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        "build",
        "dist",
        "*.egg-info",
    ]
    
    path_parts = Path(rel_path).parts
    for pattern in ignore_patterns:
        if pattern in path_parts or any(part.startswith('.') for part in path_parts[:-1]):
            return True
    return False


def scan_directory(base_path: Path, respect_gitignore: bool = True) -> Dict[str, List[Dict]]:
    """
    Scan directory for Python files and extract structure.
    
    Args:
        base_path: Root directory to scan
        respect_gitignore: Whether to respect .gitignore patterns
    
    Returns:
        Dict mapping file paths to their extracted definitions
    """
    base_path = Path(base_path).resolve()
    gitignore = load_gitignore(base_path) if respect_gitignore else None
    code_structure = {}
    
    print(f"🔍 Scanning: {base_path}")
    
    file_count = 0
    skipped_count = 0
    
    for root, dirs, files in os.walk(base_path):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if not should_ignore_path(d)]
        
        rel_root = Path(root).relative_to(base_path)
        
        for file in files:
            if not file.endswith(".py"):
                continue
            
            full_path = Path(root) / file
            rel_path = full_path.relative_to(base_path)
            
            # Check gitignore
            if gitignore and gitignore.match_file(str(rel_path)):
                skipped_count += 1
                continue
            
            # Check common ignore patterns
            if should_ignore_path(str(rel_path)):
                skipped_count += 1
                continue
            
            try:
                defs = extract_defs_from_file(full_path)
                if defs:  # Only include files with definitions
                    code_structure[str(rel_path)] = defs
                    file_count += 1
            except Exception as e:
                print(f"❌ Failed to parse {rel_path}: {e}")
    
    print(f"✅ Scanned {file_count} files ({skipped_count} skipped)")
    return code_structure


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def format_args(args: List[str]) -> str:
    """Format function arguments"""
    if not args:
        return "()"
    if args[0] == "self":
        args = args[1:]
    if not args:
        return "(self)"
    return f"(self, {', '.join(args)})" if args else "(self)"


def print_structure(structure: Dict[str, List[Dict]], 
                   output_file: Optional[Path] = None,
                   show_line_numbers: bool = False) -> None:
    """
    Print structure to console and/or file.
    
    Args:
        structure: Dictionary mapping file paths to definitions
        output_file: Optional file path to write output
        show_line_numbers: Whether to show line numbers
    """
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append("📂 PROJECT STRUCTURE")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Files: {len(structure)}")
    lines.append("=" * 80)
    lines.append("")
    
    # Sort files alphabetically
    sorted_files = sorted(structure.items())
    
    for file_path, defs in sorted_files:
        lines.append(f"📄 {file_path}")
        
        if not defs:
            lines.append("   (empty)")
            lines.append("")
            continue
        
        for item in defs:
            lineno = f":{item['lineno']}" if show_line_numbers else ""
            
            if item["type"] == "class":
                lines.append(f"   🧱 class {item['name']}{lineno}")
                
                if item["methods"]:
                    for i, method in enumerate(item["methods"]):
                        is_last = (i == len(item["methods"]) - 1)
                        prefix = "└──" if is_last else "├──"
                        args_str = format_args(method["args"])
                        method_lineno = f":{method['lineno']}" if show_line_numbers else ""
                        lines.append(f"      {prefix} def {method['name']}{args_str}{method_lineno}")
                else:
                    lines.append("      (no methods)")
            
            elif item["type"] == "function":
                args_str = format_args(item["args"])
                lines.append(f"   ⚙️  def {item['name']}{args_str}{lineno}")
        
        lines.append("")
    
    # Summary statistics
    total_classes = sum(1 for defs in structure.values() for d in defs if d["type"] == "class")
    total_functions = sum(1 for defs in structure.values() for d in defs if d["type"] == "function")
    total_methods = sum(len(d["methods"]) for defs in structure.values() for d in defs if d["type"] == "class")
    
    lines.append("=" * 80)
    lines.append("📊 STATISTICS")
    lines.append("=" * 80)
    lines.append(f"Total Classes:    {total_classes}")
    lines.append(f"Total Functions:  {total_functions}")
    lines.append(f"Total Methods:    {total_methods}")
    lines.append("=" * 80)
    
    # Join all lines
    output = "\n".join(lines)
    
    # Print to console
    print(output)
    
    # Write to file if specified
    if output_file:
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"\n✅ Output saved to: {output_file}")
        except Exception as e:
            print(f"\n❌ Failed to write to {output_file}: {e}")


def generate_tree_view(structure: Dict[str, List[Dict]]) -> str:
    """Generate a simple tree view of files"""
    lines = ["📂 Project Tree", ""]
    
    # Build directory tree
    files_by_dir = {}
    for file_path in sorted(structure.keys()):
        dir_path = str(Path(file_path).parent)
        if dir_path == ".":
            dir_path = "(root)"
        files_by_dir.setdefault(dir_path, []).append(Path(file_path).name)
    
    for dir_path, files in sorted(files_by_dir.items()):
        lines.append(f"  📁 {dir_path}")
        for file in sorted(files):
            lines.append(f"     └── 📄 {file}")
        lines.append("")
    
    return "\n".join(lines)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="Scan Python project and extract structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan current directory
  python project_scanner.py

  # Scan specific directory
  python project_scanner.py /path/to/project

  # Save to file
  python project_scanner.py --output structure.txt

  # Show line numbers
  python project_scanner.py --line-numbers

  # Generate tree view
  python project_scanner.py --tree
        """
    )
    
    parser.add_argument(
        "project_dir",
        nargs="?",
        default=".",
        help="Project directory to scan (default: current directory)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file path (e.g., structure.txt)"
    )
    parser.add_argument(
        "-l", "--line-numbers",
        action="store_true",
        help="Show line numbers for definitions"
    )
    parser.add_argument(
        "-t", "--tree",
        action="store_true",
        help="Generate simple tree view"
    )
    parser.add_argument(
        "--no-gitignore",
        action="store_true",
        help="Don't respect .gitignore patterns"
    )
    
    args = parser.parse_args()
    
    # Scan directory
    project_path = Path(args.project_dir).resolve()
    
    if not project_path.exists():
        print(f"❌ Error: Directory does not exist: {project_path}")
        return 1
    
    structure = scan_directory(
        project_path,
        respect_gitignore=not args.no_gitignore
    )
    
    if not structure:
        print("⚠️  No Python files with definitions found.")
        return 0
    
    # Generate output
    if args.tree:
        tree_output = generate_tree_view(structure)
        print("\n" + tree_output)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(tree_output)
    else:
        print_structure(
            structure,
            output_file=args.output,
            show_line_numbers=args.line_numbers
        )
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
