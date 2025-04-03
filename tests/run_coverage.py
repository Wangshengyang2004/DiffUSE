#!/usr/bin/env python
"""
Run all tests for the DiffUSE project with coverage analysis.
"""
import os
import sys
import subprocess
from pathlib import Path


def run_coverage():
    """Run all tests with coverage analysis."""
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Get the project root directory
    project_root = script_dir.parent
    
    # Run coverage
    result = subprocess.run([
        sys.executable, '-m', 'pytest',
        str(script_dir),
        '--cov=src.diffuse',
        '--cov-report=term',
        '--cov-report=html:coverage_html',
        '-v',
        '--color=yes'
    ], cwd=project_root)
    
    # Print message about coverage report
    if result.returncode == 0:
        print("\nCoverage report generated. Open coverage_html/index.html to view the report.")
    
    return result.returncode


if __name__ == '__main__':
    sys.exit(run_coverage())