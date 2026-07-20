from pathlib import Path

import yaml


def _load_yaml(relative_path: str) -> dict:
    path = Path(relative_path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_support_crew_tasks_only_reference_support_agents():
    agents = _load_yaml("agents/crews/config/support_agents.yaml")
    tasks = _load_yaml("agents/crews/config/support_tasks.yaml")

    referenced_agents = {task_config["agent"] for task_config in tasks.values()}
    # context_gatherer was removed in Phase 2: retrieval is deterministic and
    # no longer an agent task. Only the synthesizer remains.
    assert referenced_agents == {"answer_synthesizer"}
    assert referenced_agents.issubset(set(agents))


def test_support_yaml_does_not_claim_rrf_it_does_not_perform():
    """RRF used to be prose in these files instructing the model to rank data
    it was never given. It is real code now (app/services/retrieval/fusion.py),
    and the prompts must not re-acquire the claim."""
    raw = (
        Path("agents/crews/config/support_agents.yaml").read_text(encoding="utf-8")
        + Path("agents/crews/config/support_tasks.yaml").read_text(encoding="utf-8")
    )
    lowered = raw.lower()
    for phrase in ("reciprocal rank fusion", "rrf"):
        # Allowed only where the comment explains that it moved to code.
        for line in lowered.splitlines():
            if phrase in line and not line.lstrip().startswith("#"):
                raise AssertionError(
                    f"support YAML instructs the model to perform {phrase!r}; "
                    f"fusion is code, not a prompt."
                )


def test_memory_crew_tasks_only_reference_memory_agents():
    agents = _load_yaml("agents/crews/config/memory_agents.yaml")
    tasks = _load_yaml("agents/crews/config/memory_tasks.yaml")

    referenced_agents = {task_config["agent"] for task_config in tasks.values()}
    assert referenced_agents == {"entity_extractor", "graph_updater"}
    assert referenced_agents.issubset(set(agents))


def test_graph_search_import_does_not_connect():
    """Importing graph_search should NOT create a Neo4j driver at import time.

    The driver is now lazy-initialized on first use, so this import succeeds
    even when Neo4j is not running (which is the case in unit test environments).
    """
    import importlib

    import app.services.graph_search as gs

    # Force a fresh import to prove it doesn't crash
    importlib.reload(gs)

    # The private _driver should still be None after import (lazy)
    assert gs._driver is None

