"""
Notebook Service - Utilities for Jupyter notebook integration.
"""
import os
import json
from typing import Optional, Dict
from paths import NOTEBOOKS_DIR, OUTPUT_DIR, JUPYTER_URL


def get_session_notebook_path() -> str:
    """
    Get the path to the session notebook template.
    The notebook is a static file in the notebooks directory.

    Returns:
        Path to the session notebook
    """
    return os.path.join(NOTEBOOKS_DIR, "ltn_training_session.ipynb")


def session_notebook_exists() -> bool:
    """
    Check if the session notebook template exists.

    Returns:
        True if the session notebook exists
    """
    return os.path.exists(get_session_notebook_path())


def get_notebook_download_content(notebook_path: str) -> bytes:
    """
    Get notebook content as bytes for download.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        Notebook content as bytes
    """
    with open(notebook_path, 'rb') as f:
        return f.read()


def get_jupyter_notebook_url(notebook_name: str) -> str:
    """
    Get the URL to open a notebook in JupyterLab.

    Args:
        notebook_name: Name of the notebook file

    Returns:
        Full URL to open the notebook in JupyterLab
    """
    return f"{JUPYTER_URL}/lab/tree/{notebook_name}"


