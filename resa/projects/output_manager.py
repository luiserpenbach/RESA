"""
Output directory management for RESA projects.

Handles creation of timestamped outputs, multiple export formats,
and cleanup of old outputs.
"""
import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np

from resa.core.results import EngineDesignResult


def _make_serializable(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return {k: _make_serializable(v) for k, v in obj.__dict__.items()
                if not k.startswith('_')}
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


class OutputManager:
    """
    Manages output directory structure and file exports.

    Creates timestamped output files for HTML reports, CSV data,
    and JSON configurations.
    """

    DEFAULT_FORMATS = ['html', 'csv', 'json']

    def __init__(
        self,
        base_dir: str = './output',
        project_name: Optional[str] = None
    ):
        """
        Initialize output manager.

        Args:
            base_dir: Base output directory
            project_name: Optional project name for subdirectory
        """
        self.base_dir = Path(base_dir)
        self.project_name = project_name

        # Create project subdirectory if specified
        if project_name:
            self.output_dir = self.base_dir / self._sanitize_name(project_name)
        else:
            self.output_dir = self.base_dir

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize a name for use as filename/dirname."""
        # Replace problematic characters with underscores
        invalid = '<>:"/\\|?*'
        result = name
        for char in invalid:
            result = result.replace(char, '_')
        return result.strip()

    def _generate_timestamp(self) -> str:
        """Generate timestamp string for filenames."""
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    def get_output_dir(self, create: bool = True) -> str:
        """
        Get the output directory path.

        Args:
            create: If True, create the directory if it doesn't exist

        Returns:
            Path to output directory as string
        """
        if create:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        return str(self.output_dir)

    def save_result(
        self,
        result: Union[EngineDesignResult, Dict[str, Any]],
        name: str,
        formats: Optional[List[str]] = None,
        config: Optional[Any] = None
    ) -> Dict[str, str]:
        """
        Save analysis result to multiple formats.

        Args:
            result: EngineDesignResult or dictionary to save
            name: Base name for output files
            formats: List of formats ('html', 'csv', 'json')
            config: Optional config to include in outputs

        Returns:
            Dictionary mapping format to output file path
        """
        if formats is None:
            formats = self.DEFAULT_FORMATS

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = self._generate_timestamp()
        safe_name = self._sanitize_name(name)
        base_filename = f"{safe_name}_{timestamp}"

        output_paths = {}

        for fmt in formats:
            fmt = fmt.lower()
            if fmt == 'html':
                path = self._save_html(result, base_filename, config)
            elif fmt == 'csv':
                path = self._save_csv(result, base_filename)
            elif fmt == 'json':
                path = self._save_json(result, base_filename, config)
            else:
                raise ValueError(f"Unsupported format: {fmt}")

            output_paths[fmt] = str(path)

        return output_paths

    def _save_html(
        self,
        result: Union[EngineDesignResult, Dict],
        base_filename: str,
        config: Optional[Any] = None
    ) -> Path:
        """Save result as HTML report."""
        filepath = self.output_dir / f"{base_filename}.html"

        # Convert result to dictionary for display
        if hasattr(result, 'summary_dict'):
            summary = result.summary_dict()
        elif isinstance(result, dict):
            summary = result
        else:
            summary = _make_serializable(result)

        # Generate HTML
        html_content = self._generate_html_report(summary, config, base_filename)

        with open(filepath, 'w') as f:
            f.write(html_content)

        return filepath

    def _generate_html_report(
        self,
        summary: Dict[str, Any],
        config: Optional[Any],
        title: str
    ) -> str:
        """Generate HTML report content."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Build summary table rows
        summary_rows = ""
        for key, value in summary.items():
            summary_rows += f"        <tr><td>{key}</td><td>{value}</td></tr>\n"

        # Build config section if provided
        config_section = ""
        if config:
            config_dict = config.to_dict() if hasattr(config, 'to_dict') else config
            config_rows = ""
            for key, value in config_dict.items():
                config_rows += f"        <tr><td>{key}</td><td>{value}</td></tr>\n"
            config_section = f"""
    <h2>Configuration</h2>
    <table>
      <thead>
        <tr><th>Parameter</th><th>Value</th></tr>
      </thead>
      <tbody>
{config_rows}
      </tbody>
    </table>
"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RESA Analysis Report - {title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .header .timestamp {{
            opacity: 0.8;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        h2 {{
            color: #2c5282;
            margin-top: 30px;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RESA Analysis Report</h1>
        <div class="timestamp">Generated: {timestamp}</div>
    </div>

    <h2>Performance Summary</h2>
    <table>
      <thead>
        <tr><th>Metric</th><th>Value</th></tr>
      </thead>
      <tbody>
{summary_rows}
      </tbody>
    </table>
{config_section}
    <div class="footer">
        Generated by RESA - Rocket Engine Simulation & Analysis
    </div>
</body>
</html>
"""
        return html

    def _save_csv(
        self,
        result: Union[EngineDesignResult, Dict],
        base_filename: str
    ) -> Path:
        """Save result data as CSV."""
        filepath = self.output_dir / f"{base_filename}.csv"

        # Get data arrays if available
        data_to_export = {}

        if hasattr(result, 'cooling') and result.cooling:
            cooling = result.cooling
            if hasattr(cooling, 'T_coolant') and cooling.T_coolant is not None:
                data_to_export['T_coolant_K'] = cooling.T_coolant
            if hasattr(cooling, 'T_wall_hot') and cooling.T_wall_hot is not None:
                data_to_export['T_wall_hot_K'] = cooling.T_wall_hot
            if hasattr(cooling, 'T_wall_cold') and cooling.T_wall_cold is not None:
                data_to_export['T_wall_cold_K'] = cooling.T_wall_cold
            if hasattr(cooling, 'P_coolant') and cooling.P_coolant is not None:
                data_to_export['P_coolant_Pa'] = cooling.P_coolant
            if hasattr(cooling, 'q_flux') and cooling.q_flux is not None:
                data_to_export['q_flux_W_m2'] = cooling.q_flux
            if hasattr(cooling, 'velocity') and cooling.velocity is not None:
                data_to_export['velocity_m_s'] = cooling.velocity

        if hasattr(result, 'nozzle_geometry') and result.nozzle_geometry:
            geom = result.nozzle_geometry
            if hasattr(geom, 'x_full') and geom.x_full is not None:
                data_to_export['x_mm'] = geom.x_full
            if hasattr(geom, 'y_full') and geom.y_full is not None:
                data_to_export['y_mm'] = geom.y_full

        if hasattr(result, 'mach_numbers') and result.mach_numbers is not None:
            data_to_export['Mach'] = result.mach_numbers

        if hasattr(result, 'T_gas_recovery') and result.T_gas_recovery is not None:
            data_to_export['T_gas_recovery_K'] = result.T_gas_recovery

        # If no array data, export summary
        if not data_to_export:
            if hasattr(result, 'summary_dict'):
                summary = result.summary_dict()
            elif isinstance(result, dict):
                summary = result
            else:
                summary = {"result": str(result)}

            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Parameter', 'Value'])
                for key, value in summary.items():
                    writer.writerow([key, value])
        else:
            # Find max length
            max_len = max(len(arr) for arr in data_to_export.values() if hasattr(arr, '__len__'))

            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header
                writer.writerow(list(data_to_export.keys()))
                # Data rows
                for i in range(max_len):
                    row = []
                    for key, arr in data_to_export.items():
                        if hasattr(arr, '__len__') and i < len(arr):
                            row.append(arr[i])
                        else:
                            row.append('')
                    writer.writerow(row)

        return filepath

    def _save_json(
        self,
        result: Union[EngineDesignResult, Dict],
        base_filename: str,
        config: Optional[Any] = None
    ) -> Path:
        """Save result as JSON."""
        filepath = self.output_dir / f"{base_filename}.json"

        output = {
            "timestamp": datetime.now().isoformat(),
            "result": _make_serializable(result),
        }

        if config:
            output["config"] = _make_serializable(config)

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        return filepath

    def list_outputs(self) -> List[Dict[str, Any]]:
        """
        List all output files in the output directory.

        Returns:
            List of dictionaries with file information:
                - name: filename
                - path: full path
                - format: file format
                - size: file size in bytes
                - modified: modification timestamp
        """
        outputs = []

        if not self.output_dir.exists():
            return outputs

        for filepath in self.output_dir.iterdir():
            if filepath.is_file():
                stat = filepath.stat()
                ext = filepath.suffix.lower().lstrip('.')
                outputs.append({
                    'name': filepath.name,
                    'path': str(filepath),
                    'format': ext,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                })

        # Sort by modification time (newest first)
        outputs.sort(key=lambda x: x['modified'], reverse=True)

        return outputs

    def cleanup_old(self, keep_latest: int = 10) -> int:
        """
        Remove old output files, keeping only the latest N.

        Args:
            keep_latest: Number of latest files to keep (per format)

        Returns:
            Number of files removed
        """
        if not self.output_dir.exists():
            return 0

        # Group files by format
        files_by_format: Dict[str, List[Path]] = {}

        for filepath in self.output_dir.iterdir():
            if filepath.is_file():
                ext = filepath.suffix.lower()
                if ext not in files_by_format:
                    files_by_format[ext] = []
                files_by_format[ext].append(filepath)

        removed_count = 0

        for ext, files in files_by_format.items():
            # Sort by modification time (newest first)
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Remove files beyond keep_latest
            for filepath in files[keep_latest:]:
                try:
                    filepath.unlink()
                    removed_count += 1
                except OSError:
                    pass  # Skip files that can't be removed

        return removed_count

    def get_latest_output(self, format: str = 'html') -> Optional[str]:
        """
        Get the path to the most recent output of a given format.

        Args:
            format: File format to look for

        Returns:
            Path to latest file or None if not found
        """
        outputs = self.list_outputs()
        for output in outputs:
            if output['format'] == format.lower().lstrip('.'):
                return output['path']
        return None

    def create_run_directory(self, run_name: Optional[str] = None) -> Path:
        """
        Create a timestamped directory for a specific run.

        Args:
            run_name: Optional name for the run

        Returns:
            Path to the new run directory
        """
        timestamp = self._generate_timestamp()
        if run_name:
            dir_name = f"{self._sanitize_name(run_name)}_{timestamp}"
        else:
            dir_name = f"run_{timestamp}"

        run_dir = self.output_dir / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)

        return run_dir

    def clear_all(self) -> int:
        """
        Remove all files in the output directory.

        Returns:
            Number of files removed
        """
        if not self.output_dir.exists():
            return 0

        count = 0
        for filepath in self.output_dir.iterdir():
            if filepath.is_file():
                try:
                    filepath.unlink()
                    count += 1
                except OSError:
                    pass

        return count
