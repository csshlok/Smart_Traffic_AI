import os
import sys
import argparse
import fnmatch
from pathlib import Path
from datetime import datetime

def human_size(n):
    units = ["B","KB","MB","GB","TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units)-1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"

def match_any(path_rel_posix, patterns):
    # Match path and each segment against glob patterns
    if not patterns:
        return False
    if any(fnmatch.fnmatch(path_rel_posix, p) for p in patterns):
        return True
    parts = path_rel_posix.split("/")
    for i in range(1, len(parts)+1):
        sub = "/".join(parts[:i])
        if any(fnmatch.fnmatch(sub, p) for p in patterns):
            return True
    return False

def build_tree(root: Path, rel: Path, args, totals, large_files):
    node = {
        "name": rel.name if rel.name else root.resolve().name,
        "path": rel,
        "type": "dir",
        "children": []
    }
    if args.max_depth is not None and rel.as_posix().count("/") >= args.max_depth:
        return node

    full = (root / rel)
    try:
        with os.scandir(full) as it:
            entries = list(it)
    except PermissionError:
        return node

    dirs = []
    files = []
    for e in entries:
        rel_child = (rel / e.name)
        rel_posix = rel_child.as_posix()
        if match_any(rel_posix, args.exclude):
            continue
        if e.is_dir(follow_symlinks=False):
            dirs.append(e)
        else:
            files.append(e)

    dirs.sort(key=lambda d: d.name.lower())
    files.sort(key=lambda f: f.name.lower())

    # Recurse into directories
    for d in dirs:
        child = build_tree(root, rel / d.name, args, totals, large_files)
        node["children"].append(child)

    # Add files with metadata
    for f in files:
        try:
            st = f.stat()
        except FileNotFoundError:
            continue
        size = st.st_size
        mtime = datetime.fromtimestamp(st.st_mtime)
        file_node = {
            "name": f.name,
            "path": rel / f.name,
            "type": "file",
            "size": size,
            "mtime": mtime
        }
        node["children"].append(file_node)
        totals["files"] += 1
        totals["bytes"] += size
        if args.large_mb is not None and size >= args.large_mb * 1024 * 1024:
            large_files.append((file_node["path"].as_posix(), size))

    totals["dirs"] += 1
    return node

def render_tree(node, prefix="", is_last=True, lines=None, root_level=True):
    if lines is None:
        lines = []
    connector = "" if root_level else ("└── " if is_last else "├── ")
    if node["type"] == "dir":
        label = f"{node['name']}/"
        lines.append(f"{prefix}{connector}{label}")
        new_prefix = "" if root_level else prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(node["children"]):
            render_tree(child, new_prefix, i == len(node["children"]) - 1, lines, root_level=False)
    else:
        size = human_size(node["size"])
        ts = node["mtime"].strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"{prefix}{connector}{node['name']}  (size: {size}, mtime: {ts})")
    return lines

def main():
    parser = argparse.ArgumentParser(description="Generate a Markdown file tree with sizes and timestamps.")
    parser.add_argument("--root", default=".", help="Root directory to scan (default: .)")
    parser.add_argument("--out", default="FILE_TREE.md", help="Output Markdown file (default: FILE_TREE.md)")
    parser.add_argument("--exclude", action="append", default=[],
                        help="Glob pattern to exclude (repeatable). Example: --exclude .git --exclude **/__pycache__/**")
    parser.add_argument("--max-depth", type=int, default=None, help="Limit directory depth (default: no limit)")
    parser.add_argument("--large-mb", type=int, default=100,
                        help="Report files >= this size in MB in a separate section (default: 100)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        print(f"Error: root path not found or not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    # Sensible default excludes
    default_excludes = [
        ".git", ".hg", ".svn", ".DS_Store",
        "**/__pycache__/**", "**/.mypy_cache/**", "**/.pytest_cache/**",
        ".venv", "venv", "env", ".idea", ".vscode",
        "node_modules", "dist", "build"
    ]
    # Only add defaults that user didn't already specify similarly
    user_patterns = set(args.exclude or [])
    for p in default_excludes:
        if p not in user_patterns:
            args.exclude.append(p)

    totals = {"dirs": 0, "files": 0, "bytes": 0}
    large_files = []
    root_node = build_tree(root, Path(""), args, totals, large_files)

    tree_lines = render_tree(root_node)
    total_size_h = human_size(totals["bytes"])

    md = []
    md.append(f"# File Tree for {root.name}")
    md.append("")
    md.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- Root: `{root.as_posix()}`")
    md.append(f"- Total directories: {totals['dirs']}")
    md.append(f"- Total files: {totals['files']}")
    md.append(f"- Total size: {total_size_h}")
    if args.exclude:
        md.append(f"- Excludes: {', '.join(args.exclude)}")
    md.append("")
    md.append("```text")
    md.extend(tree_lines)
    md.append("```")

    if large_files:
        md.append("")
        md.append(f"## Large Files (>= {args.large_mb} MB)")
        for path, size in sorted(large_files, key=lambda x: x[1], reverse=True):
            md.append(f"- `{path}` — {human_size(size)}")

    out_path = Path(args.out)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote Markdown tree to: {out_path.resolve()}")

if __name__ == "__main__":
    main()