import threading
from typing import Dict, Any, Optional


class Profiler:

    _instance: Optional['Profiler'] = None
    _lock = threading.Lock()

    _total_inferences: int = 0
    _total_queries: int = 0
    _arm_inference_counts: Dict[str, int] = {}

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def increment_inference(cls, count: int = 1, arm_name: Optional[str] = None) -> None:
        with cls._lock:
            cls._total_inferences += count
            if arm_name:
                cls._arm_inference_counts[arm_name] = (
                    cls._arm_inference_counts.get(arm_name, 0) + count
                )

    @classmethod
    def increment_query(cls) -> None:
        with cls._lock:
            cls._total_queries += 1

    @classmethod
    def record_step_summary(cls, num_arms: int) -> None:
        with cls._lock:
            cls._total_queries += 1
            cls._total_inferences += num_arms

    @classmethod
    def get_total_cost(cls) -> int:
        return cls._total_inferences

    @classmethod
    def get_total_queries(cls) -> int:
        return cls._total_queries

    @classmethod
    def get_avg_arms_per_query(cls) -> float:
        if cls._total_queries == 0:
            return 0.0
        return cls._total_inferences / cls._total_queries

    @classmethod
    def get_arm_costs(cls) -> Dict[str, int]:
        return cls._arm_inference_counts.copy()

    @classmethod
    def get_statistics(cls) -> Dict[str, Any]:
        return {
            "total_queries": cls._total_queries,
            "total_inferences": cls._total_inferences,
            "avg_arms_per_query": cls.get_avg_arms_per_query(),
            "arm_costs": cls.get_arm_costs(),
        }

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._total_inferences = 0
            cls._total_queries = 0
            cls._arm_inference_counts = {}
