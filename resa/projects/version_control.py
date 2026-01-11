"""
Git-based version control for engine designs.

Provides versioning, tagging, and comparison of engine configurations and results.
"""
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from resa.core.config import EngineConfig
from resa.core.results import EngineDesignResult


def _serialize_value(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, datetime):
        return {"__datetime__": True, "iso": obj.isoformat()}
    elif hasattr(obj, '__dict__'):
        return {k: _serialize_value(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: _serialize_value(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_value(v) for v in obj]
    return obj


def _deserialize_value(obj: Any) -> Any:
    """Convert JSON data back to Python objects."""
    if isinstance(obj, dict):
        if obj.get("__ndarray__"):
            return np.array(obj["data"], dtype=obj.get("dtype", "float64"))
        elif obj.get("__datetime__"):
            return datetime.fromisoformat(obj["iso"])
        return {k: _deserialize_value(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deserialize_value(v) for v in obj]
    return obj


@dataclass
class DesignVersion:
    """Represents a saved version of an engine design."""

    version_id: str
    timestamp: datetime
    description: str
    author: str
    parent_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Key metrics for quick comparison
    metrics: Dict[str, float] = field(default_factory=dict)

    # Config summary (key fields)
    config_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version_id": self.version_id,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "author": self.author,
            "parent_id": self.parent_id,
            "tags": self.tags,
            "metrics": self.metrics,
            "config_summary": self.config_summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DesignVersion':
        """Create from dictionary."""
        return cls(
            version_id=data["version_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            description=data["description"],
            author=data["author"],
            parent_id=data.get("parent_id"),
            tags=data.get("tags", []),
            metrics=data.get("metrics", {}),
            config_summary=data.get("config_summary", {}),
        )

    def __repr__(self) -> str:
        tag_str = f" [{', '.join(self.tags)}]" if self.tags else ""
        return f"DesignVersion({self.version_id[:8]}... {self.timestamp.strftime('%Y-%m-%d %H:%M')}{tag_str})"


class ProjectVersionControl:
    """
    Git-inspired version control for engine design projects.

    Stores versions as JSON files with hash-based IDs.
    Tracks parent versions for history tree.
    """

    VERSIONS_DIR = "versions"
    INDEX_FILE = "versions_index.json"
    CURRENT_FILE = "current_version.txt"

    def __init__(self, project_dir: str):
        """
        Initialize version control for a project directory.

        Args:
            project_dir: Path to the project directory
        """
        self.project_dir = Path(project_dir)
        self.versions_dir = self.project_dir / self.VERSIONS_DIR
        self.index_path = self.versions_dir / self.INDEX_FILE
        self.current_path = self.versions_dir / self.CURRENT_FILE

        # Create directories if needed
        self.versions_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize index
        self._index: Dict[str, DesignVersion] = {}
        self._tags: Dict[str, str] = {}  # tag -> version_id
        self._load_index()

    def _load_index(self):
        """Load version index from disk."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                data = json.load(f)

            self._index = {
                vid: DesignVersion.from_dict(vdata)
                for vid, vdata in data.get("versions", {}).items()
            }
            self._tags = data.get("tags", {})

    def _save_index(self):
        """Save version index to disk."""
        data = {
            "versions": {vid: v.to_dict() for vid, v in self._index.items()},
            "tags": self._tags,
        }
        with open(self.index_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _generate_version_id(self, config_data: Dict, result_data: Dict, timestamp: datetime) -> str:
        """Generate a unique hash-based version ID."""
        content = json.dumps({
            "config": config_data,
            "result_metrics": {k: v for k, v in result_data.items() if isinstance(v, (int, float, str, bool))},
            "timestamp": timestamp.isoformat(),
        }, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def _extract_metrics(self, result: EngineDesignResult) -> Dict[str, float]:
        """Extract key metrics from result for quick comparison."""
        metrics = {
            "isp_vac": result.isp_vac,
            "isp_sea": result.isp_sea,
            "thrust_vac": result.thrust_vac,
            "thrust_sea": result.thrust_sea,
            "pc_bar": result.pc_bar,
            "mr": result.mr,
            "dt_mm": result.dt_mm,
            "de_mm": result.de_mm,
            "expansion_ratio": result.expansion_ratio,
        }

        if result.cooling:
            metrics["T_wall_max"] = result.cooling.max_wall_temp
            metrics["q_flux_max"] = result.cooling.max_heat_flux
            metrics["pressure_drop"] = result.cooling.pressure_drop

        return {k: float(v) if v is not None else 0.0 for k, v in metrics.items()}

    def _extract_config_summary(self, config: EngineConfig) -> Dict[str, Any]:
        """Extract key config fields for summary."""
        return {
            "engine_name": config.engine_name,
            "fuel": config.fuel,
            "oxidizer": config.oxidizer,
            "thrust_n": config.thrust_n,
            "pc_bar": config.pc_bar,
            "mr": config.mr,
            "expansion_ratio": config.expansion_ratio,
            "L_star": config.L_star,
        }

    def _get_current_version_id(self) -> Optional[str]:
        """Get the currently checked out version ID."""
        if self.current_path.exists():
            return self.current_path.read_text().strip()
        return None

    def _set_current_version(self, version_id: str):
        """Set the current version."""
        self.current_path.write_text(version_id)

    def save_version(
        self,
        config: EngineConfig,
        result: EngineDesignResult,
        description: str,
        author: str = "unknown"
    ) -> DesignVersion:
        """
        Save a new version of the design.

        Args:
            config: Engine configuration
            result: Analysis result
            description: Description of this version
            author: Author name

        Returns:
            DesignVersion object for the saved version
        """
        timestamp = datetime.now()

        # Serialize config and result
        config_data = config.to_dict()
        result_data = _serialize_value(result.__dict__)

        # Generate version ID
        version_id = self._generate_version_id(config_data, result_data, timestamp)

        # Get parent (current version before this save)
        parent_id = self._get_current_version_id()

        # Create version object
        version = DesignVersion(
            version_id=version_id,
            timestamp=timestamp,
            description=description,
            author=author,
            parent_id=parent_id,
            metrics=self._extract_metrics(result),
            config_summary=self._extract_config_summary(config),
        )

        # Save full data to version file
        version_file = self.versions_dir / f"{version_id}.json"
        version_data = {
            "version": version.to_dict(),
            "config": config_data,
            "result": result_data,
        }
        with open(version_file, 'w') as f:
            json.dump(version_data, f, indent=2)

        # Update index
        self._index[version_id] = version
        self._save_index()

        # Set as current version
        self._set_current_version(version_id)

        return version

    def load_version(self, version_id: str) -> Tuple[EngineConfig, Dict[str, Any]]:
        """
        Load a specific version.

        Args:
            version_id: The version ID to load (can be partial)

        Returns:
            Tuple of (EngineConfig, result_dict)

        Note:
            Result is returned as a dictionary because EngineDesignResult
            contains complex nested objects that require special reconstruction.
        """
        # Resolve partial version ID
        full_id = self._resolve_version_id(version_id)

        version_file = self.versions_dir / f"{full_id}.json"
        if not version_file.exists():
            raise ValueError(f"Version {version_id} not found")

        with open(version_file, 'r') as f:
            data = json.load(f)

        # Reconstruct config
        config = EngineConfig(**data["config"])

        # Deserialize result (returns dict with numpy arrays restored)
        result_data = _deserialize_value(data["result"])

        return config, result_data

    def _resolve_version_id(self, partial_id: str) -> str:
        """Resolve a partial version ID to full ID."""
        if partial_id in self._index:
            return partial_id

        # Try partial match
        matches = [vid for vid in self._index.keys() if vid.startswith(partial_id)]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            raise ValueError(f"Ambiguous version ID '{partial_id}', matches: {matches}")
        else:
            raise ValueError(f"Version ID '{partial_id}' not found")

    def list_versions(self) -> List[DesignVersion]:
        """
        List all versions, sorted by timestamp (newest first).

        Returns:
            List of DesignVersion objects
        """
        return sorted(
            self._index.values(),
            key=lambda v: v.timestamp,
            reverse=True
        )

    def diff_versions(self, v1_id: str, v2_id: str) -> Dict[str, Any]:
        """
        Compare two versions and show differences.

        Args:
            v1_id: First version ID (base)
            v2_id: Second version ID (compare)

        Returns:
            Dictionary with:
                - config_changes: Dict of changed config fields
                - metric_changes: Dict of metric differences
                - v1_summary: Summary of version 1
                - v2_summary: Summary of version 2
        """
        v1_id = self._resolve_version_id(v1_id)
        v2_id = self._resolve_version_id(v2_id)

        v1 = self._index[v1_id]
        v2 = self._index[v2_id]

        config1, _ = self.load_version(v1_id)
        config2, _ = self.load_version(v2_id)

        # Compare configs
        config_changes = {}
        c1_dict = config1.to_dict()
        c2_dict = config2.to_dict()

        all_keys = set(c1_dict.keys()) | set(c2_dict.keys())
        for key in all_keys:
            val1 = c1_dict.get(key)
            val2 = c2_dict.get(key)
            if val1 != val2:
                config_changes[key] = {"from": val1, "to": val2}

        # Compare metrics
        metric_changes = {}
        all_metric_keys = set(v1.metrics.keys()) | set(v2.metrics.keys())
        for key in all_metric_keys:
            val1 = v1.metrics.get(key, 0.0)
            val2 = v2.metrics.get(key, 0.0)
            if val1 != val2:
                delta = val2 - val1
                pct = (delta / val1 * 100) if val1 != 0 else float('inf')
                metric_changes[key] = {
                    "from": val1,
                    "to": val2,
                    "delta": delta,
                    "pct_change": pct,
                }

        return {
            "config_changes": config_changes,
            "metric_changes": metric_changes,
            "v1_summary": {
                "id": v1_id[:8],
                "description": v1.description,
                "timestamp": v1.timestamp.isoformat(),
                "author": v1.author,
            },
            "v2_summary": {
                "id": v2_id[:8],
                "description": v2.description,
                "timestamp": v2.timestamp.isoformat(),
                "author": v2.author,
            },
        }

    def tag_version(self, version_id: str, tag: str):
        """
        Add a tag to a version.

        Args:
            version_id: Version to tag
            tag: Tag name (e.g., 'baseline', 'release-1.0')
        """
        full_id = self._resolve_version_id(version_id)

        if full_id not in self._index:
            raise ValueError(f"Version {version_id} not found")

        # Update tag mapping
        self._tags[tag] = full_id

        # Update version object
        if tag not in self._index[full_id].tags:
            self._index[full_id].tags.append(tag)

        self._save_index()

    def get_version_by_tag(self, tag: str) -> DesignVersion:
        """
        Get version by tag name.

        Args:
            tag: Tag name

        Returns:
            DesignVersion object
        """
        if tag not in self._tags:
            raise ValueError(f"Tag '{tag}' not found")

        version_id = self._tags[tag]
        return self._index[version_id]

    def get_version(self, version_id: str) -> DesignVersion:
        """
        Get version metadata by ID.

        Args:
            version_id: Version ID (can be partial)

        Returns:
            DesignVersion object
        """
        full_id = self._resolve_version_id(version_id)
        return self._index[full_id]

    def get_history(self, version_id: Optional[str] = None) -> List[DesignVersion]:
        """
        Get version history (ancestors) for a version.

        Args:
            version_id: Starting version (default: current)

        Returns:
            List of versions from given version back to root
        """
        if version_id is None:
            version_id = self._get_current_version_id()
            if version_id is None:
                return []

        full_id = self._resolve_version_id(version_id)
        history = []

        current = self._index.get(full_id)
        while current is not None:
            history.append(current)
            if current.parent_id:
                current = self._index.get(current.parent_id)
            else:
                current = None

        return history

    def remove_tag(self, tag: str):
        """Remove a tag."""
        if tag in self._tags:
            version_id = self._tags[tag]
            del self._tags[tag]
            if version_id in self._index and tag in self._index[version_id].tags:
                self._index[version_id].tags.remove(tag)
            self._save_index()

    def get_tags(self) -> Dict[str, str]:
        """Get all tags and their version IDs."""
        return dict(self._tags)

    def checkout(self, version_id: str):
        """
        Set a version as the current version.

        Args:
            version_id: Version to checkout
        """
        full_id = self._resolve_version_id(version_id)
        self._set_current_version(full_id)

    def get_current(self) -> Optional[DesignVersion]:
        """Get the current version."""
        current_id = self._get_current_version_id()
        if current_id and current_id in self._index:
            return self._index[current_id]
        return None
