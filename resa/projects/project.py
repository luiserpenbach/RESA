"""
Project container for RESA engine designs.

Provides a unified interface for managing engine design projects,
including configuration, results, version control, and outputs.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from resa.core.config import EngineConfig
from resa.core.results import EngineDesignResult
from resa.projects.version_control import ProjectVersionControl, DesignVersion
from resa.projects.output_manager import OutputManager


class Project:
    """
    Container for an engine design project.

    Manages:
        - Project metadata (name, description, dates)
        - Configuration storage
        - Analysis results history
        - Version control integration
        - Output file management
    """

    PROJECT_FILE = "project.json"
    CONFIG_FILE = "current_config.json"
    RESULTS_DIR = "results"

    def __init__(
        self,
        name: str,
        directory: str,
        description: str = "",
        author: str = ""
    ):
        """
        Initialize a project.

        Args:
            name: Project name
            directory: Project directory path
            description: Optional project description
            author: Optional author name
        """
        self.name = name
        self.directory = Path(directory)
        self.description = description
        self.author = author

        self.created_at: Optional[datetime] = None
        self.modified_at: Optional[datetime] = None

        self._config: Optional[EngineConfig] = None
        self._results: List[Dict[str, Any]] = []

        # Lazy-loaded components
        self._version_control: Optional[ProjectVersionControl] = None
        self._output_manager: Optional[OutputManager] = None

    @property
    def project_file(self) -> Path:
        """Path to project metadata file."""
        return self.directory / self.PROJECT_FILE

    @property
    def config_file(self) -> Path:
        """Path to current config file."""
        return self.directory / self.CONFIG_FILE

    @property
    def results_dir(self) -> Path:
        """Path to results directory."""
        return self.directory / self.RESULTS_DIR

    def _ensure_directories(self):
        """Create project directories if needed."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save(self):
        """
        Save project metadata and current configuration to disk.
        """
        self._ensure_directories()

        now = datetime.now()
        if self.created_at is None:
            self.created_at = now
        self.modified_at = now

        # Save project metadata
        project_data = {
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "results_count": len(self._results),
        }

        with open(self.project_file, 'w') as f:
            json.dump(project_data, f, indent=2)

        # Save current config if exists
        if self._config is not None:
            config_data = self._config.to_dict()
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)

    def load(self) -> 'Project':
        """
        Load project from disk.

        Returns:
            Self for method chaining
        """
        if not self.project_file.exists():
            raise FileNotFoundError(f"Project file not found: {self.project_file}")

        # Load metadata
        with open(self.project_file, 'r') as f:
            data = json.load(f)

        self.name = data.get("name", self.name)
        self.description = data.get("description", "")
        self.author = data.get("author", "")

        if data.get("created_at"):
            self.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("modified_at"):
            self.modified_at = datetime.fromisoformat(data["modified_at"])

        # Load current config if exists
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            self._config = EngineConfig(**config_data)

        # Load results index
        self._load_results_index()

        return self

    def _load_results_index(self):
        """Load results index from results directory."""
        self._results = []

        if not self.results_dir.exists():
            return

        for filepath in self.results_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Extract summary info
                summary = {
                    "id": filepath.stem,
                    "path": str(filepath),
                    "timestamp": data.get("timestamp"),
                }

                # Get key metrics if available
                if "metrics" in data:
                    summary["metrics"] = data["metrics"]
                elif "result" in data:
                    result = data["result"]
                    summary["metrics"] = {
                        "isp_vac": result.get("isp_vac", 0),
                        "thrust_vac": result.get("thrust_vac", 0),
                        "pc_bar": result.get("pc_bar", 0),
                    }

                self._results.append(summary)
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by timestamp (newest first)
        self._results.sort(
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )

    @classmethod
    def create(
        cls,
        name: str,
        directory: str,
        description: str = "",
        author: str = "",
        initial_config: Optional[EngineConfig] = None
    ) -> 'Project':
        """
        Create a new project.

        Args:
            name: Project name
            directory: Project directory path
            description: Optional description
            author: Optional author name
            initial_config: Optional initial configuration

        Returns:
            New Project instance
        """
        project = cls(name, directory, description, author)
        project._config = initial_config
        project.save()
        return project

    @classmethod
    def open(cls, directory: str) -> 'Project':
        """
        Open an existing project.

        Args:
            directory: Project directory path

        Returns:
            Loaded Project instance
        """
        # Create minimal project to load
        project = cls("", directory)
        project.load()
        return project

    @classmethod
    def exists(cls, directory: str) -> bool:
        """Check if a project exists at the given directory."""
        return (Path(directory) / cls.PROJECT_FILE).exists()

    def get_config(self) -> Optional[EngineConfig]:
        """
        Get the current configuration.

        Returns:
            Current EngineConfig or None if not set
        """
        return self._config

    def set_config(self, config: EngineConfig):
        """
        Set the current configuration.

        Args:
            config: New configuration
        """
        self._config = config
        self.modified_at = datetime.now()

    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get list of analysis results.

        Returns:
            List of result summaries with id, path, timestamp, and metrics
        """
        return list(self._results)

    def add_result(
        self,
        result: EngineDesignResult,
        name: Optional[str] = None,
        description: str = ""
    ) -> str:
        """
        Add a new analysis result to the project.

        Args:
            result: Analysis result to save
            name: Optional name for the result
            description: Optional description

        Returns:
            ID of the saved result
        """
        self._ensure_directories()

        timestamp = datetime.now()
        result_id = timestamp.strftime('%Y%m%d_%H%M%S')
        if name:
            result_id = f"{name}_{result_id}"

        # Extract metrics
        metrics = {
            "isp_vac": result.isp_vac,
            "isp_sea": result.isp_sea,
            "thrust_vac": result.thrust_vac,
            "thrust_sea": result.thrust_sea,
            "pc_bar": result.pc_bar,
            "mr": result.mr,
        }
        if result.cooling:
            metrics["T_wall_max"] = result.cooling.max_wall_temp

        # Save result data
        result_data = {
            "id": result_id,
            "name": name or result_id,
            "description": description,
            "timestamp": timestamp.isoformat(),
            "metrics": metrics,
            "result": self._serialize_result(result),
        }

        # Include config if available
        if self._config:
            result_data["config"] = self._config.to_dict()

        filepath = self.results_dir / f"{result_id}.json"
        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)

        # Update results index
        self._results.insert(0, {
            "id": result_id,
            "path": str(filepath),
            "timestamp": timestamp.isoformat(),
            "metrics": metrics,
        })

        self.modified_at = timestamp

        return result_id

    def _serialize_result(self, result: EngineDesignResult) -> Dict[str, Any]:
        """Serialize result to JSON-compatible dict."""
        import numpy as np

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            return obj

        return convert(result.__dict__)

    def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a specific result by ID.

        Args:
            result_id: Result ID

        Returns:
            Result data dictionary or None if not found
        """
        filepath = self.results_dir / f"{result_id}.json"
        if not filepath.exists():
            # Try partial match
            matches = list(self.results_dir.glob(f"*{result_id}*.json"))
            if len(matches) == 1:
                filepath = matches[0]
            elif not matches:
                return None
            else:
                raise ValueError(f"Ambiguous result ID: {result_id}")

        with open(filepath, 'r') as f:
            return json.load(f)

    def delete_result(self, result_id: str) -> bool:
        """
        Delete a result by ID.

        Args:
            result_id: Result ID

        Returns:
            True if deleted, False if not found
        """
        filepath = self.results_dir / f"{result_id}.json"
        if filepath.exists():
            filepath.unlink()
            self._results = [r for r in self._results if r["id"] != result_id]
            return True
        return False

    def get_version_control(self) -> ProjectVersionControl:
        """
        Get the version control system for this project.

        Returns:
            ProjectVersionControl instance
        """
        if self._version_control is None:
            self._version_control = ProjectVersionControl(str(self.directory))
        return self._version_control

    def get_output_manager(self) -> OutputManager:
        """
        Get the output manager for this project.

        Returns:
            OutputManager instance
        """
        if self._output_manager is None:
            output_dir = self.directory / "output"
            self._output_manager = OutputManager(
                base_dir=str(output_dir),
                project_name=None  # Already in project directory
            )
        return self._output_manager

    def save_version(
        self,
        result: EngineDesignResult,
        description: str,
        author: Optional[str] = None
    ) -> DesignVersion:
        """
        Save current state as a new version.

        Args:
            result: Current analysis result
            description: Version description
            author: Optional author (uses project author if not specified)

        Returns:
            DesignVersion object
        """
        if self._config is None:
            raise ValueError("No configuration set. Use set_config() first.")

        vc = self.get_version_control()
        return vc.save_version(
            config=self._config,
            result=result,
            description=description,
            author=author or self.author
        )

    def load_version(self, version_id: str) -> EngineConfig:
        """
        Load a version and set it as current config.

        Args:
            version_id: Version ID to load

        Returns:
            Loaded EngineConfig
        """
        vc = self.get_version_control()
        config, _ = vc.load_version(version_id)
        self._config = config
        return config

    def export_results(
        self,
        result: EngineDesignResult,
        name: str,
        formats: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Export results using the output manager.

        Args:
            result: Analysis result to export
            name: Base name for output files
            formats: List of formats (default: ['html', 'csv', 'json'])

        Returns:
            Dictionary mapping format to output file path
        """
        om = self.get_output_manager()
        return om.save_result(
            result=result,
            name=name,
            formats=formats,
            config=self._config
        )

    def summary(self) -> Dict[str, Any]:
        """
        Get project summary.

        Returns:
            Dictionary with project information
        """
        vc = self.get_version_control()
        versions = vc.list_versions()

        return {
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "directory": str(self.directory),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "has_config": self._config is not None,
            "results_count": len(self._results),
            "versions_count": len(versions),
            "current_version": vc.get_current().version_id[:8] if vc.get_current() else None,
        }

    def __repr__(self) -> str:
        return f"Project('{self.name}', directory='{self.directory}')"
