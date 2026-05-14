"""
Unit tests for application settings — no model artifacts required.

These tests always run in CI. They verify that Settings loads with sensible
defaults and that env var overrides work correctly.
"""



class TestSettingsDefaults:
    def test_api_version(self):
        from src.config import settings
        assert settings.api_version == "0.1.0"

    def test_log_level_uppercase(self):
        from src.config import settings
        assert settings.log_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        assert settings.log_level == settings.log_level.upper()

    def test_outputs_dir_name(self):
        from src.config import settings
        assert settings.outputs_dir.name == "outputs"

    def test_processed_dir_name(self):
        from src.config import settings
        assert settings.processed_dir.name == "processed"

    def test_outputs_is_child_of_data_root(self):
        from src.config import settings
        assert settings.outputs_dir.parent == settings.data_root

    def test_api_title_non_empty(self):
        from src.config import settings
        assert len(settings.api_title) > 0

    def test_api_description_non_empty(self):
        from src.config import settings
        assert len(settings.api_description) > 0


class TestSettingsEnvOverride:
    def test_log_level_override(self, monkeypatch):
        monkeypatch.setenv("PRICING_LOG_LEVEL", "debug")
        # Re-import to pick up env change (module-level singleton won't re-evaluate,
        # so we instantiate a fresh Settings directly).
        from src.config import Settings
        s = Settings()
        assert s.log_level == "DEBUG"  # validator upcases

    def test_data_root_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PRICING_DATA_ROOT", str(tmp_path))
        from src.config import Settings
        s = Settings()
        assert s.data_root == tmp_path.resolve()
        assert s.outputs_dir == tmp_path.resolve() / "outputs"
