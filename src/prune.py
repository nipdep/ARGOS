import uuid
import re
from sqlglot import parse_one, exp
from sqlglot.expressions import Expression, Identifier, Column

from src.data.ASTTree import TreeNode
from src.operators.astObject import SqlglotOperator
from src.operators.astTree import ASTTreeOperator
from src.operators.ontologyInstance import OntologyOperator


class ASTPruner:
    """
    Prunes a sqlglot AST based on the state of an associated ontology.
    """
    base_clause = {
            "SelectStatement": "SelectClause",
            "UpdateStatement": "UpdateClause",
            "InsertStatement": "InsertClause",
            "DeleteStatement": "DeleteClause",
        }

    def __init__(self, onto_op: OntologyOperator):
        self.onto_op = onto_op
        self.sql_op = onto_op.tree_op.sql_op
        self.tree_op = onto_op.tree_op
        self.violated_ids = set()

        
    def prune(self):
        """
        Orchestrates the full, multi-stage pruning process.
        The underlying AST is modified in place.
        """
        print("--- Starting AST Pruning Process ---")
        print(f"Query before pruning: {self.sql_op.ast}")

        self.column_ref_instances = self.onto_op.get_column_ref_instances()
        self.table_ref_instances = self.onto_op.get_table_ref_instances()

        self._prune_from_table_status()
        if self.sql_op.ast is not None:
            # If the AST is None, it means the root statement was violated.
            # No need to proceed further.
            # print(f"    - SQL AST after Stage 1: {self.sql_op.ast}")
            self._prune_from_column_status()
        
        # After all modifications, rebuild the custom tree to reflect changes
        # print("--> Pruning complete. Rebuilding custom tree...")
        # self.tree_op.build_tree()
        # print("--- Pruning Process Finished ---")

    import re

    def normalize_condition(self, condition: str) -> str:
        """
        Cleans up a single SQL condition string to make it comparable.
        - Converts to lowercase
        - Removes extra whitespace
        - Removes whitespace around operators
        """
        # Convert to lowercase and strip leading/trailing whitespace
        norm = condition.strip().lower()
        # Replace multiple spaces with a single space
        norm = re.sub(r'\s+', ' ', norm)
        # Remove whitespace around operators like =, >, <, etc.
        norm = re.sub(r'\s*([=<>!]+)\s*', r'\1', norm)
        return norm

    def sub_condition_exists_no_parser(self, full_condition: str, sub_condition: str) -> bool:
        """
        Checks if a sub-condition exists in a full condition without parsing SQL.

        This version will strip table aliases (e.g., "t1.") from the full_condition
        before comparison, allowing "col = 1" to match "t1.col = 1".

        LIMITATIONS:
        - Only works reliably for conditions joined by AND.
        - Does not understand OR logic or complex nested parentheses.
        - Does not understand SQL semantics (e.g., BETWEEN).
        """
        if not sub_condition:
            return True
        if not full_condition:
            return False

        try:
            # Split the full condition, strip table aliases, and normalize each part.
            full_parts = {
                # This regex removes patterns like "alias." from the condition parts.
                self.normalize_condition(re.sub(r'\b[a-zA-Z0-9_]+\.', '', part))
                for part in re.split(r'\s+AND\s+', full_condition, flags=re.IGNORECASE)
            }

            # Split the sub-condition by 'AND' and normalize each part.
            sub_parts = {
                self.normalize_condition(part)
                for part in re.split(r'\s+AND\s+', sub_condition, flags=re.IGNORECASE)
            }

            # Check if the set of sub-condition parts is a subset of the full condition parts.
            return sub_parts.issubset(full_parts)
        except Exception:
            # If any regex or other error occurs, assume failure.
            return False

    def _prune_from_table_status(self):
        """
        Stage 1: Handles removals and modifications based on TableRef statuses.
        """
        print("--> Stage 1: Pruning based on table statuses...")
        self.violated_ids = []
        self.rowtag_ids = []
        view_denied_table_entity_ids = set()
        process_denied_table_entity_ids = set()
        process_denied_table_ref_ids = set()

        for inst, attr in self.table_ref_instances.items():
            status = attr['Status']
            # Condition A.1: Violated Table -> Remove Statement
            if status == "Violated":
                parent_clause = self.tree_op.get_parent_clause(inst)
                table_ref_node = self.tree_op.get_node_by_id(inst)
                table_entity_id = getattr(table_ref_node, "table_reference_id", None) if table_ref_node else None
                scopes = self._resolve_table_violation_scopes(inst, attr)

                if parent_clause is None:
                    continue

                print(f" Parent clause name: {parent_clause.name} and kind: {parent_clause.kind} | Instance: {inst}")
                # Scope-aware behavior:
                # - table VIEW deny => prune only projection columns from that table
                # - table PROCESS deny => prune non-projection/process columns from that table
                # - unknown scope (legacy) => fallback to full table trace removal
                if scopes:
                    if table_entity_id:
                        if "view" in scopes:
                            view_denied_table_entity_ids.add(table_entity_id)
                        if "process" in scopes:
                            process_denied_table_entity_ids.add(table_entity_id)
                            process_denied_table_ref_ids.add(inst)
                    continue

                parent_stmt = self.tree_op.get_parent_statement(inst)
                if parent_stmt:
                    print(f">> Instance: {inst}")
                    if parent_stmt.id == self.tree_op.root.id:
                        print(f">>> Instance: {inst}")
                        self.sql_op.ast = None
                        return
                    self.violated_ids.append(parent_stmt.id)

                    base_clause_type = ASTPruner.base_clause.get(parent_stmt.name, None)
                    clause_id = next((c.id for c in parent_stmt.children if c.name == base_clause_type), None)
                    if clause_id:
                        self.violated_ids.append(clause_id)

            # Condition A.2: RowTag Table -> Add WHERE condition
            elif status == "RowTag":
                self.rowtag_ids.append(inst)

        if view_denied_table_entity_ids:
            self._mark_columns_for_violated_tables(
                view_denied_table_entity_ids,
                violation_scope="view",
            )
        if process_denied_table_entity_ids:
            self._mark_columns_for_violated_tables(
                process_denied_table_entity_ids,
                # For table-level process violations (e.g., rtp), remove all
                # references to the denied table in any query position.
                violation_scope="all",
            )
        if process_denied_table_ref_ids:
            self.violated_ids.extend(sorted(process_denied_table_ref_ids))
        try:
            # Step 3: Apply the main pruning transformer with all collected IDs
            self.sql_op.ast = self.sql_op.ast.transform(self._pruning_transformer)
        except AssertionError as e:
                self.sql_op.ast = None
                return
        # except Exception as e:
        #     print(f"[Error] Failed to apply pruning transformer: {e}")
        #     return
        # self.sql_op.ast = self.sql_op.ast.transform(self._pruning_transformer)
        # print(f" Row Tag IDs: {self.rowtag_ids}")
        for inst in self.rowtag_ids:
            # print(f"    - Processing RowTag TableRef: {inst}")
            self._add_rowtag_condition(inst)

    def _mark_columns_for_violated_tables(
        self,
        violated_table_entity_ids: set[str],
        violation_scope: str = "all",
    ) -> None:
        """
        Collect ColumnRef node IDs that reference violated tables.

        `violation_scope`:
        - "view": remove projection references (SelectClause)
        - "process": remove non-projection references (legacy behavior)
        - "top_level_projection": remove only references in the root statement's SelectClause
        - "all": remove all references
        """
        for column_ref_id in self.column_ref_instances.keys():
            column_ref_node = self.tree_op.get_node_by_id(column_ref_id)
            if not column_ref_node:
                continue
            table_entity_id = getattr(column_ref_node, "table_reference_id", None)
            if table_entity_id not in violated_table_entity_ids:
                continue
            if not self._column_matches_violation_scope(column_ref_id, violation_scope):
                continue
            self.violated_ids.append(column_ref_id)

    def _column_matches_violation_scope(self, column_ref_id: str, violation_scope: str) -> bool:
        if violation_scope == "all":
            return True

        parent_clause = self.tree_op.get_parent_clause(column_ref_id)
        clause_name = parent_clause.name if parent_clause else ""
        is_projection = clause_name == "SelectClause"

        if violation_scope == "view":
            return is_projection
        if violation_scope == "top_level_projection":
            if not is_projection:
                return False
            parent_statement = self.tree_op.get_parent_statement(column_ref_id)
            return bool(
                self.tree_op.root
                and parent_statement
                and parent_statement.id == self.tree_op.root.id
            )
        if violation_scope == "process":
            return not is_projection
        return True

    def _resolve_table_violation_scopes(self, table_ref_id: str, table_attr: dict) -> set[str]:
        """
        Resolve violated table scope(s) from related policies/rules.

        Returns a subset of {"view", "process"}.
        """
        scopes: set[str] = set()

        node = None
        if hasattr(self.onto_op, "get_instance_by_id"):
            node = self.onto_op.get_instance_by_id(table_ref_id)

        if node is not None:
            for policy_obj in getattr(node, "relatedPolicy", []) or []:
                for action_scope in getattr(policy_obj, "hasActionScope", []) or []:
                    scope_name = str(getattr(action_scope, "name", action_scope)).strip().lower()
                    if "view" in scope_name:
                        scopes.add("view")
                    elif "process" in scope_name:
                        scopes.add("process")

        if scopes:
            return scopes

        # Fallbacks for partial metadata/exported status maps.
        policy_name = str((table_attr or {}).get("Policy") or "").strip().lower()
        rule_name = str((table_attr or {}).get("Rule") or "").strip().lower()

        if "view" in policy_name:
            scopes.add("view")
        if "process" in policy_name:
            scopes.add("process")

        # Rule-name fallback (for split table-read rules).
        if rule_name == "rtv":
            scopes.add("view")
        if rule_name == "rtp":
            scopes.add("process")

        return scopes

    # def _add_rowtag_condition(self, inst: str):
    #     print(f"    - TableRef '{inst}' is RowTag. Adding condition to parent statement.")
    #     policy_list = self.onto_op.get_node_attribute(inst, "relatedPolicy")
    #     # print(f"    - Found related policies: {policy_list}")
    #     if not policy_list: return
        
    #     # get data property 'hasCondition' from the first policy
    #     policy_id = str(policy_list).strip("[]").split(".")[-1]
    #     # print(f"    - Using policy ID: {policy_id}, {policy_list[0]}")
    #     condition_list = self.onto_op.get_node_attribute(policy_id, "hasActionCondition")
    #     print(f"    - Found conditions: {condition_list}")
    #     if not condition_list: return
        
    #     condition_str = str(condition_list).strip("['']").strip('"')
    #     # print(repr(condition_str))
    #     parent_stmt = self.tree_op.get_parent_statement(inst)
    #     where_cls = self.tree_op.get_where_clauses(parent_stmt)
    #     # print(f" Where clause: {where_cls} | {len(where_cls)}")
    #     if len(where_cls):
    #         where_cls = where_cls[0]  # Assuming we only care about the first WHERE clause
    #         where_cls_str = where_cls.get_value()
    #         # print(f"where clauses: {where_cls_str} | {condition_str} | {self.sub_condition_exists_no_parser(where_cls_str, condition_str)}")
    #         # print(f"parent statement: {parent_stmt}")
    #         if not(self.sub_condition_exists_no_parser(where_cls_str, condition_str)) and parent_stmt.name == "SelectStatement":
    #             # print(`f"    - TableRef '{inst}' has RowTag. Adding condition to statement '{parent_stmt.id[:8]}': {condition_str}")
    #             parent_id = "n" + parent_stmt.id[1:]
    #             stmt_sqlglot_node = self.sql_op.get_node_by_id(parent_id)
    #             # print(f"    - SQLGlot node for statement: {stmt_sqlglot_node} | {parent_stmt.id}")
    #             if stmt_sqlglot_node:
    #                 # print(f"stmt_sqlglot_node: {stmt_sqlglot_node} | {stmt_sqlglot_node.where}")
    #                 stmt_sqlglot_node.where(condition_str, copy=False)
    #     else:
    #         print(f"    - No WHERE clause found for statement '{parent_stmt.id[:8]}'. Adding new condition.")
    #         parent_id = "n" + parent_stmt.id[1:]
    #         stmt_sqlglot_node = self.sql_op.get_node_by_id(parent_id)
    #         # print(f"    - SQLGlot node for statement: {stmt_sqlglot_node} | {parent_stmt.id}")
    #         if stmt_sqlglot_node:
    #             print("Running ")
    #             # print(f"stmt_sqlglot_node: {stmt_sqlglot_node} | {stmt_sqlglot_node.where}")
    #             stmt_sqlglot_node.where(condition_str, copy=False)

    def _add_rowtag_condition(self, inst: str):
        print(f"     - TableRef '{inst}' is RowTag. Adding condition to parent statement.")
        policy_list = self.onto_op.get_node_attribute(inst, "relatedPolicy")
        if not policy_list: return
        
        policy_id = str(policy_list).strip("[]").split(".")[-1]
        condition_list = self.onto_op.get_node_attribute(policy_id, "hasActionCondition")
        print(f"     - Found conditions: {condition_list}")
        if not condition_list: return
        
        condition_str = str(condition_list).strip("['']").strip('"')
        parent_stmt = self.tree_op.get_parent_statement(inst)
        
        # We need the ID of the statement we want to modify
        parent_id = "n" + parent_stmt.id[1:]

        # Define a small "transformer" function that will be applied to every node in the tree.
        def add_where_clause(node):
            # Check if the current node is the one we want to modify.
            # We check the type and the ID to be safe.
            if isinstance(node, exp.Select) and node.meta.get('id') == parent_id:
                print(f"     - Found target statement '{parent_id[:8]}'. Applying WHERE clause.")
                
                # Use the immutable .where() method to create the new, modified node.
                # This works whether a WHERE clause exists or not.
                new_node = node.where(condition_str, copy=False)
                
                # Return the new node to replace the old one in the tree.
                return new_node
            
            # For all other nodes, return them unchanged.
            return node

        # --- THE FIX ---
        # Call .transform() on the main AST. It will walk the tree, apply our function,
        # and return a completely new, correct AST.
        # We then reassign self.sql_op.ast to this new tree.
        where_cls = self.tree_op.get_where_clauses(parent_stmt)
        if where_cls:
            where_cls_str = where_cls[0].get_value()
            print(f"     - Where clauses: {where_cls_str} | Condition: {condition_str} | Exists: {self.sub_condition_exists_no_parser(where_cls_str, condition_str)}")
            if not(self.sub_condition_exists_no_parser(where_cls_str, condition_str)):
                self.sql_op.ast = self.sql_op.ast.transform(add_where_clause)
        else:
            self.sql_op.ast = self.sql_op.ast.transform(add_where_clause)

    def _prune_from_column_status(self):
        """
        Stage 2: Handles cascading removals based on ColumnRef 'Violated' status.
        This runs iteratively to ensure all cascading effects are resolved.
        """
        print("--> Stage 2: Pruning based on column statuses...")
        # Step 1: Collect initial violated IDs and propagate them
        self.violated_ids += [id for id, status in self.column_ref_instances.items() if status['Status'] == "Violated"]
        print(f"     - Initial violated column IDs: {self.violated_ids}")
        if not self.violated_ids:
            print("     - No violated columns found.")
            return
        
        # print(f"     - Initial violated column IDs: {self.violated_ids}")
        self._propagate_violations_through_aliases()

        # Step 2: Iteratively apply the pruning transformer until the AST is stable
        max_iterations = 10  # Safeguard against potential infinite loops
        for i in range(max_iterations):
            previous_sql = self.sql_op.ast.sql()
            
            # Apply the main pruning transformer
            # print(f"     - Iteration {i + 1}: Applying pruning transformer... {self.sql_op.ast}")
            try:
                self.sql_op.ast = self.sql_op.ast.transform(self._pruning_transformer)
            except AssertionError as e:
                self.sql_op.ast = None
                break

            # current_sql = self.sql_op.ast.sql(pretty=True)

            # If the AST has not changed, the pruning is complete
            if not self.sql_op.ast or self.sql_op.ast.sql() == previous_sql:
                print(f"    - Pruning complete. AST stabilized after {i + 1} iteration(s).")
                break
            
            if i == max_iterations - 1:
                print("     - Warning: Pruning reached max iterations. The AST may not be stable.")
        else:
            # This part runs if the loop finishes without a 'break'
            # which indicates it hit the max_iterations limit.
            print("     - Pruning complete (max iterations reached).")

    def _remove_node_by_id(self, target_id: str):
        """
        Removes a node from the SQLGlot AST and the custom tree by its ID.
        This is a utility method to encapsulate the removal logic.
        """
        # print(f"Removing node with ID: {target_id}")
        
        # Remove from the custom tree
        self.tree_op.remove_node_by_id(target_id)

        # Also remove from the SQLGlot AST
        self.sql_op.remove_node_by_id(target_id)

    def _propagate_violations_through_aliases(self):
        """Finds columns that are aliased and marks their usages as violated."""
        # This is a simplified propagation. A full implementation for complex
        # nested queries would require more sophisticated scope analysis.
        tainted_aliases = set()
        for alias_node in self.sql_op.ast.find_all(exp.Alias):
            print(f" Alias node: {alias_node}")
            source_col_id = alias_node.this.meta.get('id')
            if source_col_id in self.violated_ids:
                tainted_aliases.add(alias_node.alias)
        
        if tainted_aliases:
            # print(f"    - Propagating violations from tainted aliases: {tainted_aliases}")
            for col_node in self.sql_op.ast.find_all(exp.Column):
                if col_node.this.name in tainted_aliases:
                    self.violated_ids.add(col_node.meta.get('id'))

    def _pruning_transformer(self, node):
        """
        The core transformation logic for cascading removals, using the correct
        sqlglot expression classes.
        """
        # Base Case: If the node's ID is marked for removal, remove it.
        if hasattr(node, 'meta') and node.meta.get('id') in self.violated_ids:
            return None

        # Rule 1: Handle logical connectors (AND, OR).
        # This is the corrected class name.
        if isinstance(node, exp.Connector):
            left_is_gone = node.left is None
            right_is_gone = node.right is None

            if left_is_gone and not right_is_gone:
                return node.right  # Promote the survivor
            if right_is_gone and not left_is_gone:
                return node.left   # Promote the survivor
            if left_is_gone and right_is_gone:
                return None        # Both sides gone, remove the operator

        # Rule 2: Handle all other binary operations (Comparisons, Arithmetic).
        elif isinstance(node, exp.Binary):
            if node.left is None or node.right is None:
                return None

        # Rule 2.1: Clean up unary wrappers (e.g., NOT x) when x is removed.
        if isinstance(node, exp.Unary) and getattr(node, "this", None) is None:
            return None

        # Rule 2.2: Clean up IS predicates if either side is removed.
        if isinstance(node, exp.Is):
            if node.this is None or node.expression is None:
                return None

        # Rule 3: Clean up a WHERE clause that is now empty.
        if isinstance(node, exp.Where) and (node.this is None):
            return None

        # Rule 4: Clean up a SELECT statement that has no columns left.
        if isinstance(node, exp.Select):
            # Repair source clauses after table pruning. If FROM vanished but
            # there is an allowed join source, promote it into FROM.
            self._repair_select_sources(node)
            if not node.expressions:
                # print("     - Cascading removal of SELECT statement with no columns.")
                return None
        
        # Rule 5: Clean up a JOIN if its ON condition is invalid.
        if isinstance(node, exp.Join):
            # Remove JOIN only when its table source is gone.
            if node.this is None:
                return None
            # If ON became invalid due pruned operands, downgrade to a plain join
            # without ON (comma/cross semantics) so allowed table traces survive.
            if isinstance(node.on, exp.Binary) and (node.on.left is None or node.on.right is None):
                node.set("on", None)

        # Rule 6: Clean up a GROUP BY clause that has no columns left.
        if isinstance(node, exp.Group) and not node.expressions:
            return None

        # Rule 7 (NEW): Clean up a FROM clause if no source can be repaired.
        if isinstance(node, exp.From) and node.this is None:
            return None

        # Rule 8: clean up invalid function nodes.
        # Note: many valid function-like expressions (e.g., searched CASE)
        # intentionally have `this is None`, so we must only remove a function
        # when one of its required args is missing.
        if isinstance(node, exp.Func) and self._is_invalid_function_node(node):
            print(
                f"    - Cascading removal of Function '{node.sql()}' "
                "because a required argument was removed."
            )
            return None
        
        if isinstance(node, exp.Alias) and node.this is None:
            print(f"    - Cascading removal of Alias '{node.sql()}' because its expression was removed.")
            return None
        
        # Rule 9: Handle IN expressions if either side is removed. 
        elif isinstance(node, exp.In):
            # Always check the left side first.
            if node.this is None:
                return None

            # Helper function to check if an item from the right side is pruned.
            # It's pruned if it's None OR if it's a Subquery wrapper whose content is None.
            def is_pruned(item):
                return item is None or (isinstance(item, exp.Subquery) and item.this is None)

            # Case A: Handle the subquery stored in the internal `args['query']`.
            if 'query' in node.args:
                if is_pruned(node.args.get('query')):
                    return None
            
            # Case B: Handle standard value lists or parsed subqueries in `.expressions`.
            else:
                if not node.expressions or all(is_pruned(e) for e in node.expressions):
                    return None

        return node

    @staticmethod
    def _repair_select_sources(node: exp.Select) -> None:
        """
        Ensure SELECT keeps a valid source after table-pruning.

        If FROM source is removed but JOIN sources remain, promote the first
        surviving JOIN source into FROM and keep the rest as JOINs.
        """
        joins = [j for j in (node.args.get("joins") or []) if j is not None]
        if not joins and node.args.get("from") is not None:
            return

        from_clause = node.args.get("from")
        from_this = getattr(from_clause, "this", None) if from_clause is not None else None

        if from_this is not None:
            return

        promoted_source = None
        remaining_joins = []
        for join in joins:
            join_this = getattr(join, "this", None)
            if promoted_source is None and join_this is not None:
                promoted_source = join_this
                continue
            remaining_joins.append(join)

        if promoted_source is None:
            return

        if from_clause is None:
            from_clause = exp.From(this=promoted_source)
            node.set("from", from_clause)
        else:
            from_clause.set("this", promoted_source)
        node.set("joins", remaining_joins)

    @staticmethod
    def _is_invalid_function_node(node: exp.Func) -> bool:
        """
        Returns True when a function expression is structurally invalid.

        sqlglot encodes each function class with `arg_types`, where required
        arguments are marked as True. We only prune when those required args
        are missing after prior pruning steps.
        """
        # COUNT has optional arg_types in sqlglot, but COUNT(DISTINCT col)
        # can degrade into COUNT(DISTINCT) after pruning and must be removed.
        if isinstance(node, exp.Count):
            count_arg = node.args.get("this")
            count_expressions = node.args.get("expressions")
            has_star = isinstance(count_arg, exp.Star) or any(
                isinstance(e, exp.Star) for e in (count_expressions or [])
            )
            if has_star:
                return False
            if (
                ASTPruner._is_effectively_missing_arg(count_arg)
                and ASTPruner._is_effectively_missing_arg(count_expressions)
            ):
                return True

        arg_types = getattr(node, "arg_types", {}) or {}
        for arg_name, is_required in arg_types.items():
            if not is_required:
                continue
            value = node.args.get(arg_name)
            if ASTPruner._is_effectively_missing_arg(value):
                return True
        return False

    @staticmethod
    def _is_effectively_missing_arg(value) -> bool:
        """
        Returns True when an argument is semantically empty after pruning.

        Example: COUNT(DISTINCT col) can become COUNT(DISTINCT) when `col`
        is removed. sqlglot keeps this as a Distinct wrapper with empty
        expressions, so a plain None-check is not enough.
        """
        if value is None:
            return True
        if isinstance(value, list):
            return (not value) or all(ASTPruner._is_effectively_missing_arg(v) for v in value)
        if isinstance(value, exp.Distinct):
            return ASTPruner._is_effectively_missing_arg(value.expressions)
        return False
