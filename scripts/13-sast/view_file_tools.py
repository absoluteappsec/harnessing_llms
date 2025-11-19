"""
File viewing tools for LangGraph ReAct demonstration.
Simplified versions of file tools for educational purposes.
"""

import os
import re
from typing import Optional, Type
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class ViewFileInput(BaseModel):
    filepath: str = Field(description="Path to the file to view")


class ViewFileTool(BaseTool):
    name: str = "view_file"
    description: str = "Views the contents of a specified file"
    args_schema: Type[ViewFileInput] = ViewFileInput

    # Keep files reasonable for demo purposes
    MAX_FILE_SIZE_BYTES: int = 100_000  # 100KB

    def _run(
        self, filepath: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        # Clean up the filepath from any LLM artifacts
        cleaned_filepath = re.sub(
            r"^\s*```(?:json|python|text|sh|bash|plaintext)?\s*|\s*```\s*$",
            "",
            filepath.strip(),
            flags=re.DOTALL,
        )
        cleaned_filepath = re.sub(r"[\n`]", "", cleaned_filepath)
        cleaned_filepath = cleaned_filepath.strip()

        # Normalize the path
        try:
            normalized_filepath = os.path.abspath(os.path.expanduser(cleaned_filepath))
        except Exception as e:
            return f"[Error]: Could not normalize path '{cleaned_filepath}': {e}"

        # Check if file exists
        if not os.path.exists(normalized_filepath):
            return f"[Error]: File does not exist: {normalized_filepath}"
        if not os.path.isfile(normalized_filepath):
            return f"[Error]: Path exists but is not a file: {normalized_filepath}"

        # Check file size
        try:
            file_size = os.path.getsize(normalized_filepath)
            if file_size > self.MAX_FILE_SIZE_BYTES:
                return f"[Error]: File is too large ({file_size} bytes > {self.MAX_FILE_SIZE_BYTES} bytes). Use view_file_lines for large files."

            # Read and return file contents
            with open(normalized_filepath, "r", encoding="utf-8", errors="replace") as file:
                content = file.read()
                return f"File: {normalized_filepath}\nSize: {file_size} bytes\n\nContent:\n{content}"

        except Exception as e:
            return f"[Error]: Failed to read file '{normalized_filepath}': {e}"


class ViewFileLinesInput(BaseModel):
    filepath: str = Field(description="Path to the file to view")
    start_line: int = Field(description="Starting line number (1-indexed)")
    end_line: int = Field(description="Ending line number (1-indexed, inclusive)")


class ViewFileLinesTool(BaseTool):
    name: str = "view_file_lines"
    description: str = "Views specific line ranges of a file. Use for large files or specific sections."
    args_schema: Type[ViewFileLinesInput] = ViewFileLinesInput

    def _run(
        self,
        filepath: str,
        start_line: int,
        end_line: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # Clean up the filepath
        cleaned_filepath = re.sub(
            r"^\s*```(?:json|python|text|sh|bash|plaintext)?\s*|\s*```\s*$",
            "",
            filepath.strip(),
            flags=re.DOTALL,
        )
        cleaned_filepath = re.sub(r"[\n`]", "", cleaned_filepath)
        cleaned_filepath = cleaned_filepath.strip()

        # Normalize the path
        try:
            normalized_filepath = os.path.abspath(os.path.expanduser(cleaned_filepath))
        except Exception as e:
            return f"[Error]: Could not normalize path '{cleaned_filepath}': {e}"

        # Check if file exists
        if not os.path.exists(normalized_filepath):
            return f"[Error]: File does not exist: {normalized_filepath}"
        if not os.path.isfile(normalized_filepath):
            return f"[Error]: Path exists but is not a file: {normalized_filepath}"

        # Validate line numbers
        if start_line < 1:
            return f"[Error]: Start line must be >= 1, got {start_line}"
        if end_line < start_line:
            return f"[Error]: End line ({end_line}) must be >= start line ({start_line})"

        # Limit lines per request for demo
        max_lines_per_request = 100
        if (end_line - start_line + 1) > max_lines_per_request:
            return f"[Error]: Too many lines requested ({end_line - start_line + 1}). Maximum {max_lines_per_request} lines per request."

        try:
            with open(normalized_filepath, "r", encoding="utf-8", errors="replace") as file:
                lines = file.readlines()
                total_lines = len(lines)

                # Check if requested lines exist
                if start_line > total_lines:
                    return f"[Error]: Start line {start_line} exceeds file length ({total_lines} lines)"

                # Adjust end_line if it exceeds file length
                actual_end_line = min(end_line, total_lines)

                # Extract the requested lines (convert to 0-indexed)
                selected_lines = lines[start_line - 1 : actual_end_line]

                # Format the output with line numbers
                result = f"File: {normalized_filepath}\n"
                result += f"Lines {start_line}-{actual_end_line} of {total_lines}:\n\n"

                for i, line in enumerate(selected_lines, start=start_line):
                    line_content = line.rstrip("\n")
                    result += f"{i:4d}: {line_content}\n"

                return result

        except Exception as e:
            return f"[Error]: Failed to read file '{normalized_filepath}': {e}"
