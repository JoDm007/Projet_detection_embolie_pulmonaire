# backend/tests/test_jobs.py — bloc services : garde-fou mémoire de la file d'analyses
from types import SimpleNamespace

import pytest

from backend.engines.base import EngineUnavailable
from backend.services import jobs as jobs_module
from backend.services.jobs import JobManager


def _fake_ram(monkeypatch, available_gb):
    monkeypatch.setattr(jobs_module.psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=available_gb * 2**30))


def test_analysis_refused_when_ram_too_low(settings, monkeypatch):
    manager = JobManager(settings, engines={})
    _fake_ram(monkeypatch, 1.5)
    with pytest.raises(EngineUnavailable, match="Mémoire insuffisante"):
        manager._check_free_memory("test")
    manager.shutdown()


def test_analysis_allowed_with_enough_ram(settings, monkeypatch):
    manager = JobManager(settings, engines={})
    _fake_ram(monkeypatch, settings.min_free_ram_gb + 1)
    manager._check_free_memory("test")
    manager.shutdown()
