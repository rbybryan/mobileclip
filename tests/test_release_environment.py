"""The optional shared-machine setup must not guess a private server path."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_user_site_requires_an_explicit_location():
    env = {k: v for k, v in os.environ.items() if k != "MVC_PYTHONUSERBASE"}
    result = subprocess.run(["bash", "-c", "source experiments/env_b100.sh"],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Set MVC_PYTHONUSERBASE" in result.stderr


def test_explicit_user_site_and_data_overrides_are_preserved(tmp_path):
    env = os.environ.copy()
    env.update(MVC_PYTHONUSERBASE=str(tmp_path / "private user site"),
               MVC_DATA_DIR=str(tmp_path / "data"), MVC_ANNO_DIR=str(tmp_path / "annotations"))
    command = ('source experiments/env_b100.sh >/dev/null; '
               'test "$PYTHONUSERBASE" = "$MVC_PYTHONUSERBASE" && '
               'test "$MVC_DATA_DIR" = "$EXPECTED_DATA" && '
               'test "$MVC_ANNO_DIR" = "$EXPECTED_ANNO"')
    env.update(EXPECTED_DATA=env["MVC_DATA_DIR"], EXPECTED_ANNO=env["MVC_ANNO_DIR"])
    subprocess.run(["bash", "-c", command], cwd=ROOT, env=env, check=True)
