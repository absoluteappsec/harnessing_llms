"""
Directory viewing tools for LangGraph ReAct demonstration.
Simplified versions of directory tools for educational purposes.
"""

import os
import re
from typing import Optional, Type
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class DirectoryListingInput(BaseModel):
    directory: str = Field(description="Directory path to list contents from")


class DirectoryListingTool(BaseTool):
    name: str = "list_directories"
    description: str = "Lists subdirectories in the specified directory"
    args_schema: Type[DirectoryListingInput] = DirectoryListingInput

    def _run(
        self, directory: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        # Clean up the directory path from any LLM artifacts
        cleaned_directory = re.sub(
            r"^\s*```(?:json|python|text|sh|bash|plaintext)?\s*|\s*```\s*$",
            "",
            directory.strip(),
            flags=re.DOTALL,
        )
        cleaned_directory = re.sub(r"[\n`]", "", cleaned_directory)
        cleaned_directory = cleaned_directory.strip()

        # Normalize the path
        try:
            normalized_directory = os.path.abspath(os.path.expanduser(cleaned_directory))
        except Exception as e:
            return f"[Error]: Could not normalize path '{cleaned_directory}': {e}"

        # Check if directory exists
        if not os.path.exists(normalized_directory):
            return f"[Error]: Directory does not exist: {normalized_directory}"
        if not os.path.isdir(normalized_directory):
            return f"[Error]: Path exists but is not a directory: {normalized_directory}"

        try:
            # Directories to exclude (common ones that clutter output)
            exclude_dirs = [
                "venv", "env", "node_modules", ".git", "__pycache__", 
                ".venv", "vtm_venv", ".pytest_cache"
            ]
            max_depth = 2  # Limit recursion depth for demo

            directories = []
            base_depth = normalized_directory.count(os.sep)

            # Add the root directory itself first
            directories.append(f"Root Directory: {normalized_directory}")

            for root, dirs, _ in os.walk(normalized_directory):
                # Calculate current depth
                current_depth = root.count(os.sep) - base_depth
                if current_depth >= max_depth:
                    # Clear dirs list to prevent further recursion
                    dirs.clear()
                    continue

                # Filter out excluded directories
                dirs[:] = [d for d in dirs if d not in exclude_dirs]

                for dir_name in dirs:
                    full_path = os.path.join(root, dir_name)
                    if os.path.isdir(full_path):
                        directories.append(f"Directory: {full_path}")

            if len(directories) <= 1:  # Only the root directory was found
                return f"Only found the root directory: {normalized_directory}. No subdirectories found."

            return "\n".join(directories)
        except Exception as e:
            return f"[Error]: Failed to list directory: {str(e)}"


class FileListingInput(BaseModel):
    directory: str = Field(description="Directory path to list file contents from")


class FileListingTool(BaseTool):
    name: str = "list_files"
    description: str = "Lists all files in the specified directory (non-recursive)"
    args_schema: Type[FileListingInput] = FileListingInput

    def _run(
        self, directory: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        # Clean up the directory path
        cleaned_directory = re.sub(
            r"^\s*```(?:json|python|text|sh|bash|plaintext)?\s*|\s*```\s*$",
            "",
            directory.strip(),
            flags=re.DOTALL,
        )
        cleaned_directory = re.sub(r"[\n`]", "", cleaned_directory)
        cleaned_directory = cleaned_directory.strip()

        # Normalize the path
        try:
            normalized_directory = os.path.abspath(os.path.expanduser(cleaned_directory))
        except Exception as e:
            return f"[Error]: Could not normalize path '{cleaned_directory}': {e}"

        # Check if directory exists
        if not os.path.exists(normalized_directory):
            return f"[Error]: Directory does not exist: {normalized_directory}"
        if not os.path.isdir(normalized_directory):
            return f"[Error]: Path exists but is not a directory: {normalized_directory}"

        try:
            # List only files in the specified directory (non-recursive)
            files = []
            max_files = 50  # Limit number of files for demo

            for item in os.listdir(normalized_directory):
                if len(files) >= max_files:
                    files.append(f"... (more files exist but limited to {max_files} for display)")
                    break

                item_path = os.path.join(normalized_directory, item)
                if os.path.isfile(item_path):
                    # Get file size for additional info
                    try:
                        file_size = os.path.getsize(item_path)
                        files.append(f"File: {item_path} ({file_size} bytes)")
                    except:
                        files.append(f"File: {item_path}")

            if not files:
                return f"No files found in {normalized_directory}"

            return f"Files in {normalized_directory}:\n" + "\n".join(files)
        except Exception as e:
            return f"[Error]: Failed to list files: {str(e)}"


class DirectoryStructureTool(BaseTool):
    name: str = "show_directory_structure"
    description: str = "Shows a tree-like structure of directories and files (limited depth for readability)"
    args_schema: Type[DirectoryListingInput] = DirectoryListingInput

    def _run(
        self, directory: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        # Clean up the directory path
        cleaned_directory = re.sub(
            r"^\s*```(?:json|python|text|sh|bash|plaintext)?\s*|\s*```\s*$",
            "",
            directory.strip(),
            flags=re.DOTALL,
        )
        cleaned_directory = re.sub(r"[\n`]", "", cleaned_directory)
        cleaned_directory = cleaned_directory.strip()

        # Normalize the path
        try:
            normalized_directory = os.path.abspath(os.path.expanduser(cleaned_directory))
        except Exception as e:
            return f"[Error]: Could not normalize path '{cleaned_directory}': {e}"

        # Check if directory exists
        if not os.path.exists(normalized_directory):
            return f"[Error]: Directory does not exist: {normalized_directory}"
        if not os.path.isdir(normalized_directory):
            return f"[Error]: Path exists but is not a directory: {normalized_directory}"

        def build_tree(path, prefix="", max_depth=2, current_depth=0):
            if current_depth >= max_depth:
                return []
            
            exclude_items = [
                "venv", "env", "node_modules", ".git", "__pycache__", 
                ".venv", "vtm_venv", ".pytest_cache", ".DS_Store"
            ]
            
            try:
                items = sorted(os.listdir(path))
                items = [item for item in items if item not in exclude_items]
                
                tree_lines = []
                for i, item in enumerate(items):
                    item_path = os.path.join(path, item)
                    is_last = i == len(items) - 1
                    
                    if is_last:
                        current_prefix = prefix + "└── "
                        next_prefix = prefix + "    "
                    else:
                        current_prefix = prefix + "├── "
                        next_prefix = prefix + "│   "
                    
                    if os.path.isdir(item_path):
                        tree_lines.append(f"{current_prefix}{item}/")
                        tree_lines.extend(build_tree(item_path, next_prefix, max_depth, current_depth + 1))
                    else:
                        try:
                            file_size = os.path.getsize(item_path)
                            tree_lines.append(f"{current_prefix}{item} ({file_size} bytes)")
                        except:
                            tree_lines.append(f"{current_prefix}{item}")
                
                return tree_lines
            except Exception as e:
                return [f"{prefix}[Error reading directory: {e}]"]

        try:
            tree_lines = [f"{normalized_directory}/"]
            tree_lines.extend(build_tree(normalized_directory))
            return "\n".join(tree_lines)
        except Exception as e:
            return f"[Error]: Failed to build directory tree: {str(e)}"
