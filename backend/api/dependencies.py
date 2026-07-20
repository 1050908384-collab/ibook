from functools import lru_cache

from orchestrator.harness import Harness


@lru_cache(maxsize=1)
def get_harness() -> Harness:
    return Harness()
