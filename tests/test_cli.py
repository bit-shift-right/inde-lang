import sys

import pytest

from inde.inde import main


def test_help_displays_usage(capsys, monkeypatch):
    for flag in ["-h", "--help"]:
        monkeypatch.setattr(sys, "argv", ["inde", flag])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "usage: inde" in captured.out


def test_version_displays_version(capsys, monkeypatch):
    for flag in ["-V", "--version"]:
        monkeypatch.setattr(sys, "argv", ["inde", flag])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "inde version" in captured.out
