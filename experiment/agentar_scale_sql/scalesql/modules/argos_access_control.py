from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


class ArgosAccessController:
    """
    ARGOS access-control runtime wrapper.

    This controller does not generate SQL. It refines an already-generated SQL
    query using ontology-based reasoning + pruning.
    """

    def __init__(
        self,
        benchmark_root: str,
        db_id: str,
        ontology_path: str,
        save_db_abox: bool = True,
        db_abox_output_path: Optional[str] = None,
    ):
        try:
            from src.argos_abox_operator import ArgosABoxOperator  # local import for optional dependency handling
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ARGOS dependencies are missing. Install required packages "
                "(e.g., rdflib, owlready2, sqlglot) to use argos_access_control."
            ) from exc

        self.benchmark_root = str(benchmark_root)
        self.db_id = str(db_id)
        self.ontology_path = str(ontology_path)
        self.db_abox_output_path = (
            Path(db_abox_output_path)
            if db_abox_output_path
            else Path(self.benchmark_root) / self.db_id / "argos_db_abox.rdf"
        )

        self.operator = ArgosABoxOperator(self.ontology_path)
        self.operator.load_database_context(Path(self.benchmark_root) / self.db_id)
        if save_db_abox:
            self.operator.save_db_abox(self.db_abox_output_path, include_tbox=False)
        self.operator.prepare_reasoner()
        self._run_warmup_query()

    @classmethod
    def from_benchmark_db(
        cls,
        benchmark_root: str,
        db_id: str,
        ontology_path: str,
        save_db_abox: bool = True,
        db_abox_output_path: Optional[str] = None,
    ) -> "ArgosAccessController":
        return cls(
            benchmark_root=benchmark_root,
            db_id=db_id,
            ontology_path=ontology_path,
            save_db_abox=save_db_abox,
            db_abox_output_path=db_abox_output_path,
        )

    def close(self) -> None:
        self.operator.close()

    def _run_warmup_query(self) -> None:
        """
        Warm reasoner on controller initialization without recreating self.operator.
        """
        warmup_table_name = getattr(self.operator, "warmup_table_name", None)
        if not warmup_table_name:
            return

        warmup_query = f'SELECT * FROM "{warmup_table_name}";'
        try:
            self.refine_query(
                role="admin",
                sql_query=warmup_query,
            )
        except Exception:
            # Warmup should not block controller initialization.
            return

    def refine_query(self, role: str, sql_query: Optional[str]) -> Dict[str, Any]:
        sql_text = (sql_query or "").strip()
        if not sql_text:
            return {
                "status": "deny",
                "error": "Missing SQL query",
                "role": role,
                "db_id": self.db_id,
                "refined_query": "",
                "invalid_refined_query": "",
                "active_policies": [],
                "active_rules": [],
                "table_status": {},
                "column_status": {},
            }

        try:
            result = self.operator.evaluate_query(
                sql_query=sql_text,
                role=role,
                cleanup_instances=True,
            )
            refined_query = result.refined_query or ""
            invalid_refined_query = ""
            validation_error = ""
            if refined_query.strip():
                try:
                    from sqlglot import parse_one

                    parse_one(refined_query)
                except Exception as exc:
                    invalid_refined_query = refined_query
                    refined_query = ""
                    validation_error = f"Invalid refined SQL: {exc}"

            return {
                "status": "allow" if refined_query.strip() else "deny",
                "error": validation_error,
                "role": role,
                "db_id": self.db_id,
                "refined_query": refined_query,
                "invalid_refined_query": invalid_refined_query,
                "active_policies": list(result.active_policies),
                "active_rules": list(result.active_rules),
                "table_status": dict(result.table_status),
                "column_status": dict(result.column_status),
            }
        except Exception as exc:
            return {
                "status": "deny",
                "error": str(exc),
                "role": role,
                "db_id": self.db_id,
                "refined_query": "",
                "invalid_refined_query": "",
                "active_policies": [],
                "active_rules": [],
                "table_status": {},
                "column_status": {},
            }


def run_argos_access_control_case(
    controller: ArgosAccessController,
    role: str,
    sql_query: Optional[str],
) -> Dict[str, object]:
    sql_text = sql_query or ""
    decision = controller.refine_query(role=role, sql_query=sql_text)
    return {
        "final_query": decision.get("refined_query", ""),
        "answer_metadata": {
            "query": sql_text,
            "refined_query": decision.get("refined_query", ""),
            "invalid_refined_query": decision.get("invalid_refined_query", ""),
            "argos_access_control_status": decision.get("status", "deny"),
            "argos_access_control_error": decision.get("error", ""),
            "role": role,
            "db_id": decision.get("db_id", controller.db_id),
            "active_policies": decision.get("active_policies", []),
            "active_rules": decision.get("active_rules", []),
            "table_status": decision.get("table_status", {}),
            "column_status": decision.get("column_status", {}),
        },
    }
