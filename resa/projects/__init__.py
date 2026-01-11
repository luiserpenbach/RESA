"""
RESA Project Management Module.

Provides tools for managing engine design projects including:
    - Version control for design iterations
    - Output file management
    - Project organization

Example usage:
    from resa.projects import Project, ProjectVersionControl, OutputManager

    # Create a new project
    project = Project.create(
        name="Hopper Engine",
        directory="./projects/hopper",
        author="Engineering Team"
    )

    # Set configuration and run analysis
    project.set_config(config)
    result = engine.analyze(config)

    # Save a version
    version = project.save_version(result, "Initial baseline design")

    # Export results
    paths = project.export_results(result, "baseline")

    # Later: load and compare versions
    versions = project.get_version_control().list_versions()
    diff = project.get_version_control().diff_versions(v1_id, v2_id)
"""

from resa.projects.version_control import (
    ProjectVersionControl,
    DesignVersion,
)

from resa.projects.output_manager import OutputManager

from resa.projects.project import Project


__all__ = [
    # Main project class
    'Project',

    # Version control
    'ProjectVersionControl',
    'DesignVersion',

    # Output management
    'OutputManager',
]
