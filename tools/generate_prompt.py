"""
This script generates a markdown file containing the contents of all relevant
files in a codebase, formatted for use as a prompt for a Large Language Model
(LLM). It recursively scans the codebase, excluding common Python-related
directories and files (e.g., __pycache__, .mypy_cache), and includes file
contents with appropriate syntax highlighting based on file extensions.
"""

import os


def get_language(filename):
    if filename == ".env.example":
        return "ini"
    ext = os.path.splitext(filename)[1].lower()
    languages = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".json": "json",
        ".md": "markdown",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".ini": "ini",
        ".txt": "text",
        ".sh": "bash",
        ".bat": "batch",
        ".sql": "sql",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rb": "ruby",
        ".php": "php",
        ".rs": "rust",
    }
    return languages.get(ext, "text")


OUTPUT_FILE = "codebase.md"
SCRIPT_PATH = "tools/generate_prompt.py"
ignored_dirs = {
    "__pycache__",
    ".mypy_cache",
    ".git",
    "node_modules",
    ".next",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
}
ignored_files = {OUTPUT_FILE, SCRIPT_PATH, ".gitignore"}
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(
        "Give me the entire file(s) back with NO changes except for what "
        "changes are necessary to execute my request. Only give me back "
        "files with changes.\n\n"
    )

    # Collect and sort all files, excluding ignored directories and files
    all_files = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [
            d
            for d in dirs
            if d not in ignored_dirs and not d.endswith(".egg-info")
        ]
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, ".").replace("\\", "/")
            if rel_path in ignored_files:
                continue  # Skip ignored files
            all_files.append(rel_path)
    all_files.sort()

    for rel_path in all_files:
        try:
            with open(rel_path, "r", encoding="utf-8") as code_file:
                content = code_file.read()
        except UnicodeDecodeError:
            # Skip binary files or handle differently
            f.write(f"{rel_path} (binary or non-UTF-8 file skipped)\n\n")
            continue
        lang = get_language(os.path.basename(rel_path))
        f.write(f"{rel_path}\n")
        f.write(f"```{lang}\n")
        f.write(content)
        f.write("\n```\n\n")
