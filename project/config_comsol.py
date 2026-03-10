from project.config import Config, MaterialConfig


def create_comsol_baseline_config() -> Config:
    """Return a Config initialized with COMSOL baseline material values."""
    return Config(material=MaterialConfig())
