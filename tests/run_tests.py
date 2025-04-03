#!/usr/bin/env python
"""
Run all tests for the DiffUSE project.
"""
import os
import sys
import pytest
from pathlib import Path


def run_tests():
    """Run all tests in the tests directory."""
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Get the project root directory
    project_root = script_dir.parent
    
    # Add project root to path so we can import from src
    sys.path.insert(0, str(project_root))
    
    # Run all tests with pytest
    exit_code = pytest.main([
        str(script_dir),
        '-v',
        '--color=yes',
        '--no-header'
    ])
    
    return exit_code


if __name__ == '__main__':
    sys.exit(run_tests()) 