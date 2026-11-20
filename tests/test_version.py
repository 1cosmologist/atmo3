from importlib.resources import files

import atmo3


def test_version_matches_installed_version_file():
    assert atmo3.__version__ == files("atmo3").joinpath("VERSION").read_text(encoding="utf-8").strip()
