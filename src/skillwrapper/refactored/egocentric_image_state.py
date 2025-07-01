"""Define classes to represent egocentric image-based environment states."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnnotatedImage:
    """An image of the environment with an (optional) associated natural language description."""

    image_path: Path  # Filepath to the image
    description: str | None  # Optional description of the photo of the environment


class EgocentricImageState:
    """An environment state represented as a collection of egocentric images."""

    def __init__(self, initial_images: dict[str, AnnotatedImage]) -> None:
        """Initialize the egocentric image-based state."""
        self.latest_images = initial_images  # Map from location names to their annotated images

    @classmethod
    def from_yaml(cls, yaml_data: dict[str, Any]) -> EgocentricImageState:
        """Import an EgocentricImageState instance from YAML data.

        :param yaml_data: Dictionary of data describing an egocentric image-based state
        :return: Constructed EgocentricImageState instance
        """
        locations: dict[str, AnnotatedImage] = {}  # Maps each location name to its image
        for location_name, image_data in yaml_data.items():
            if "image_path" not in image_data:
                raise KeyError(f"Location '{location_name}' didn't specify an 'image_path' key.")

            image_path = Path(image_data["image_path"])
            if not image_path.exists():
                error = f"Location {location_name} had invalid image path: {image_path}"
                raise FileNotFoundError(error)

            locations[location_name] = AnnotatedImage(image_path, image_data.get("description"))

        return EgocentricImageState(locations)
