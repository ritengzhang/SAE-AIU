#!/usr/bin/env python3
"""Find and clean all cache directories."""

import os
import shutil
from pathlib import Path

def find_and_clean_cache():
    """Find and clean all cache directories."""
    # Use current working directory or environment variable for project root
    base_dir = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))
    home_dir = Path.home()
    
    print(f"\n{'='*60}")
    print("Finding and cleaning cache directories")
    print(f"{'='*60}\n")
    
    found_dirs = []
    found_files = []
    
    # Find directories in project
    for pattern in ["wandb", ".wandb", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints"]:
        for path in base_dir.rglob(pattern):
            if path.is_dir():
                found_dirs.append(path)
    
    # Check common cache locations
    common_cache_dirs = [
        home_dir / ".cache" / "huggingface",
        home_dir / ".cache" / "torch",
        home_dir / ".cache" / "transformers",
        base_dir / "wandb",
        base_dir / ".wandb",
    ]
    
    for cache_dir in common_cache_dirs:
        if cache_dir.exists() and cache_dir.is_dir():
            found_dirs.append(cache_dir)
    
    # Find cache files
    for path in base_dir.rglob("*.pyc"):
        if path.is_file():
            found_files.append(path)
    for path in base_dir.rglob("*.pyo"):
        if path.is_file():
            found_files.append(path)
    
    print(f"Found {len(found_dirs)} cache directories and {len(found_files)} cache files\n")
    
    # Show what will be deleted
    if found_dirs:
        print("Cache directories to delete:")
        for d in found_dirs[:20]:  # Show first 20
            try:
                size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                print(f"  {d} ({size_mb:.1f} MB)")
            except:
                print(f"  {d}")
        if len(found_dirs) > 20:
            print(f"  ... and {len(found_dirs) - 20} more")
        print()
    
    # Delete directories
    deleted_dirs = 0
    deleted_size = 0
    for d in found_dirs:
        try:
            # Calculate size before deletion
            size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
            deleted_size += size
            
            shutil.rmtree(d)
            deleted_dirs += 1
            print(f"✓ Deleted: {d}")
        except Exception as e:
            print(f"✗ Error deleting {d}: {e}")
    
    # Delete files
    deleted_files = 0
    for f in found_files:
        try:
            size = f.stat().st_size
            deleted_size += size
            f.unlink()
            deleted_files += 1
        except Exception as e:
            print(f"✗ Error deleting {f}: {e}")
    
    print(f"\n{'='*60}")
    print("Cleanup Summary")
    print(f"{'='*60}")
    print(f"Deleted {deleted_dirs} directories")
    print(f"Deleted {deleted_files} files")
    print(f"Total space freed: {deleted_size / (1024**3):.2f} GB")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    find_and_clean_cache()
