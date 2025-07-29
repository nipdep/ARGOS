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

        for inst, attr in self.table_ref_instances.items():
            status = attr['Status']
            # Condition A.1: Violated Table -> Remove Statement
            if status == "Violated":
                parent_clause = self.tree_op.get_parent_clause(inst)
                # print(f" Parent clause name: {parent_clause.name} and kind: {parent_clause.kind}")
                if parent_clause.name != "JoinClause":
                    parent_stmt = self.tree_op.get_parent_statement(inst)
                    if parent_stmt:
                        if parent_stmt.id == self.tree_op.root.id:
                            self.sql_op.ast = None
                            # print(f"    - Root statement '{parent_stmt.id[:8]}' is violated. Removing entire AST.")
                            return  # No need to continue, the entire AST is removed.
                    # print(f"    - TableRef '{inst}' is Violated. Marking statement '{parent_stmt.id[:8]}' for removal.")
                    base_clause_type = ASTPruner.base_clause.get(parent_stmt.name, None)
                    clause_id = next((c.id for c in parent_stmt.children if c.name == base_clause_type), None)
                    # print(f"    - Base clause ID: {clause_id}")
                    if clause_id:
                        self.violated_ids.append(clause_id)

            # Condition A.2: RowTag Table -> Add WHERE condition
            elif status == "RowTag":
                # print(f"    - TableRef '{inst}' is RowTag. Adding condition to parent statement.")
                policy_list = self.onto_op.get_node_attribute(inst, "relatedPolicy")
                # print(f"    - Found related policies: {policy_list}")
                if not policy_list: continue
                
                # get data property 'hasCondition' from the first policy
                policy_id = str(policy_list).strip("[]").split(".")[-1]
                # print(f"    - Using policy ID: {policy_id}, {policy_list[0]}")
                condition_list = self.onto_op.get_node_attribute(policy_id, "hasActionCondition")
                # print(f"    - Found conditions: {condition_list}")
                if not condition_list: continue
                
                condition_str = str(condition_list).strip("['']").strip('"')
                # print(repr(condition_str))
                parent_stmt = self.tree_op.get_parent_statement(inst)
                where_cls = self.tree_op.get_where_clauses(parent_stmt)
                # print(f" Where clause: {where_cls} | {len(where_cls)}")
                if len(where_cls):
                    where_cls = where_cls[0]  # Assuming we only care about the first WHERE clause
                    where_cls_str = where_cls.get_value()
                    # print(f"where clauses: {where_cls_str} | {condition_str} | {self.sub_condition_exists_no_parser(where_cls_str, condition_str)}")
                    # print(f"parent statement: {parent_stmt}")
                    if not(self.sub_condition_exists_no_parser(where_cls_str, condition_str)) and parent_stmt.name == "SelectStatement":
                        # print(f"    - TableRef '{inst}' has RowTag. Adding condition to statement '{parent_stmt.id[:8]}': {condition_str}")
                        parent_id = "n" + parent_stmt.id[1:]
                        stmt_sqlglot_node = self.sql_op.get_node_by_id(parent_id)
                        # print(f"    - SQLGlot node for statement: {stmt_sqlglot_node} | {parent_stmt.id}")
                        if stmt_sqlglot_node:
                            # print(f"stmt_sqlglot_node: {stmt_sqlglot_node} | {stmt_sqlglot_node.where}")
                            stmt_sqlglot_node.where(condition_str, copy=False)
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

    def _prune_from_column_status(self):
        """
        Stage 2: Handles cascading removals based on ColumnRef 'Violated' status.
        This runs iteratively to ensure all cascading effects are resolved.
        """
        print("--> Stage 2: Pruning based on column statuses...")
        # Step 1: Collect initial violated IDs and propagate them
        self.violated_ids = [id for id, status in self.column_ref_instances.items() if status['Status'] == "Violated"]
        print(f"     - Initial violated column IDs: {self.violated_ids}")
        if not self.violated_ids:
            print("     - No violated columns found.")
            return
        
        # print(f"     - Initial violated column IDs: {self.violated_ids}")
        self._propagate_violations_through_aliases()

        # Step 2: Iteratively apply the pruning transformer until the AST is stable
        max_iterations = 5  # Safeguard against potential infinite loops
        for i in range(max_iterations):
            previous_sql = self.sql_op.ast.sql(pretty=True)
            
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

        # Rule 3: Clean up a WHERE clause that is now empty.
        if isinstance(node, exp.Where) and (node.this is None):
            return None

        # Rule 4: Clean up a SELECT statement that has no columns left.
        if isinstance(node, exp.Select) and not node.expressions:
            # print("     - Cascading removal of SELECT statement with no columns.")
            return None

        # Rule 5: Clean up a JOIN if its ON condition is invalid.
        if isinstance(node, exp.Join):
            # If the ON clause was completely removed OR if it's a binary expression
            # (like a = b) where one side has been pruned, remove the JOIN.
            if node.on is None or (isinstance(node.on, exp.Binary) and (node.on.left is None or node.on.right is None)):
                return None
        
        # Rule 5: Clean up a GROUP BY clause that has no columns left.
        if isinstance(node, exp.Group) and not node.expressions:
            return None

        # Rule 5: Other clean-up rules.
        if isinstance(node, exp.Join) and node.on is None:
            return None
        
        # Rule 6 (NEW): Clean up a FROM clause if its table/join has been removed.
        if isinstance(node, exp.From) and node.this is None:
            return None

        # Rule 6: cleanup empty Function node.
        if isinstance(node, exp.Func) and node.this is None:
            print(f"    - Cascading removal of Function '{node.sql()}' because its argument was removed.")
            return None
        
        if isinstance(node, exp.Alias) and node.this is None:
            print(f"    - Cascading removal of Alias '{node.sql()}' because its expression was removed.")
            return None
        
        # Rule 7: Handle IN expressions if either side is removed.
        elif isinstance(node, exp.In):
            # If the expression being checked is gone, or the list of values is empty.
            if node.this is None or not node.expressions:
                return None

        return node