
import uuid
from sqlglot import parse_one, exp
from sqlglot.expressions import Expression, Identifier, Column
from typing import List, Optional

from src.operators.astObject import SqlglotOperator
from src.data.ASTTree import TreeNode

class ASTTreeOperator:
    """
    Builds and operates on a simplified, custom tree structure derived
    from a sqlglot AST, making it easier to apply domain-specific logic.
    """
    # --- Type Mapper (integrated as a static method) ---
    CLAUSE_TYPES = {
        "From": "FromClause",
        "Group": "GroupByClause",
        "Into": "IntoClause",
        "Limit": "LimitClause",
        "Select": "SelectClause",
        "Set": "SetClause",
        "Update": "UpdateClause",
        "Values": "ValuesClause",
        "Where": "WhereClause",
        "Order": "OrderByClause",
        "Having": "HavingClause",
        "Join": "JoinClause",
        "Insert": "InsertClause",
        "Delete": "DeleteClause"
    }

    STATEMENT_TYPES = {
        "Delete": "DeleteStatement",
        "Insert": "InsertStatement",
        "Update": "UpdateStatement",
        "Select": "SelectStatement"
    }
    
    EXPRESSION_CATEGORIES = {
        "Alias": "Alias", 
        "Table": "TableRef", 
        "Column": "ColumnRef", 
        "Star": "Wildcard",
        "Identifier": "Identifier", 
        "Sum": "Function", 
        "Count": "Function", 
        "Avg": "Function",
        "Max": "Function", 
        "Min": "Function", 
        "And": "Operator", 
        "Or": "Operator", 
        "EQ": "Operator",
        "GT": "Operator", 
        "LT": "Operator", 
        "GTE": "Operator", 
        "LTE": "Operator", 
        "Literal": "Literal"
    }

 
    def map_node_type(self, node: Expression, statement=False):
        """Maps a sqlglot AST node to a simplified (Kind, Name) tuple."""
        class_name = type(node).__name__
        if statement:
            if class_name in ASTTreeOperator.STATEMENT_TYPES:
                return "Statement", ASTTreeOperator.STATEMENT_TYPES[class_name]
            else:
                return "Statement", class_name
        else:
            if class_name in ASTTreeOperator.CLAUSE_TYPES:
                return "Clause", ASTTreeOperator.CLAUSE_TYPES[class_name]
            elif class_name in ASTTreeOperator.EXPRESSION_CATEGORIES:
                return "Expression", ASTTreeOperator.EXPRESSION_CATEGORIES[class_name]
            # Default fallback
            else:
                return "Expression", class_name

    def __init__(self, sql_operator: SqlglotOperator):
        """
        Initializes the tree operator using a pre-configured SqlglotOperator.
        """
        self.sql_op = sql_operator
        self.root: TreeNode | None = None
        self.id_to_node_map: dict[str, TreeNode] = {}
        if self.sql_op and self.sql_op.ast:
            self.build_tree(self.sql_op.ast)

    def create_node(self, sqlglot_node, parent_node, statement=False) -> 'TreeNode':
        kind, name = self.map_node_type(sqlglot_node, statement=statement)
        node = TreeNode(sqlglot_node, kind=kind, name=name, parent=parent_node)
        self.id_to_node_map[node.id] = node
        # if parent_node:
        #     parent_node.add_child(node)
        return node

    def create_column_node(self, parent_sqlglot_node, parent_node, node_id: str, **kwargs) -> 'TreeNode':
        """
        Creates a new sqlglot Column node and wraps it in a TreeNode.
        """
        column_name = kwargs.get("refcol")
        table_name = kwargs.get("reftable")

        if not column_name:
            raise ValueError("A 'this' keyword argument with the column name is required.")

        # 1. Create the Identifier for the column name itself.
        # exp.to_identifier() is a safe way to handle this.
        column_identifier = exp.to_identifier(column_name)
        
        # 2. Prepare the arguments for the main Column expression.
        column_args = {"this": column_identifier}
        
        # 3. If a table name is provided, create its Identifier and add it.
        if table_name:
            table_identifier = exp.to_identifier(table_name)
            column_args["table"] = table_identifier

        # 4. Create the final sqlglot Column node with the prepared arguments.
        new_sqlglot_node = exp.Column(**column_args)

        self.sql_op.add_node(parent_sqlglot_node.meta['id'], new_sqlglot_node, arg_name="expressions")
        
        # 5. Decorate the new node with the provided ID.
        new_sqlglot_node.meta['id'] = node_id
        
        # 6. Wrap the new sqlglot node in your custom TreeNode.
        # The parent is None as it has not been inserted into the main tree yet.
        # create a new TreeNode for the custom tree
        # new_tree_node = TreeNode(new_sqlglot_node, parent=parent_node)
        # self.id_to_node_map[new_tree_node.id] = new_tree_node
        new_tree_node = self.create_node(new_sqlglot_node, parent_node)
        self.sql_op.id_to_node_map[new_tree_node.id] = new_tree_node
        parent_node.add_child(new_tree_node)
        return new_tree_node
    
    def remove_node_by_id(self, target_id: str):
        """
        Removes a single node from the custom tree based on its unique ID.
        The internal AST is updated with the result.
        """
        if not self.root or not target_id:
            return
        
        # Find the node in the custom tree
        node_to_remove = self.get_node_by_id(target_id)
        if not node_to_remove:
            print(f"[Error] Node with ID '{target_id}' not found.")
            return
        
        # Remove from the parent
        if node_to_remove.parent:
            node_to_remove.parent.remove_child(node_to_remove)

        # Remove from the id map
        del self.id_to_node_map[node_to_remove.id]

        # Also remove from the sqlglot AST
        self.sql_op.remove_node_by_id(target_id)

    def build_tree(self, sqlglot_node, parent=None):
        # print(f"Building tree for node: {repr(sqlglot_node)} | Parent: {parent}")
        if parent is not None:
            # clause_root = TreeNode(sqlglot_node, parent=parent)
            # self.id_to_node_map[clause_root.id] = clause_root
            clause_root = self.create_node(sqlglot_node, parent)
            root_node = parent
            root_node.add_child(clause_root)
        else:
            root_node = self.create_node(sqlglot_node, None, statement=True)
            # root_node = TreeNode(sqlglot_node)
            # self.id_to_node_map[root_node.id] = root_node
            self.root = root_node
            clause_root = self.create_node(sqlglot_node, root_node)
            # clause_root = TreeNode(sqlglot_node, parent=root_node)
            # self.id_to_node_map[clause_root.id] = clause_root
            root_node.add_child(clause_root)
        

        found_first_clause = False
        for key, child in sqlglot_node.args.items():
            if child is None:
                continue

            children = child if isinstance(child, list) else [child]

            for expr in children:
                if not isinstance(expr, exp.Expression): # did not understand the logic of the check
                    continue

                c_kind, c_name = self.map_node_type(expr)

                if not found_first_clause and c_kind != "Clause":
                    sub = self._build_recursive(expr, clause_root)
                    if sub:
                        clause_root.add_child(sub)
                else:
                    found_first_clause = True
                    clause_node = self.create_node(expr, root_node)
                    # clause_node = TreeNode(expr, parent=root_node)
                    # self.id_to_node_map[clause_node.id] = clause_node
                    root_node.add_child(clause_node)

                    for _, grandchild in expr.args.items():
                        if isinstance(grandchild, exp.Expression):
                            if isinstance(expr, exp.Table) and isinstance(grandchild, exp.Identifier): # this is a weak check, so try to transfer this is recursion
                                continue  # Skip Identifier under Table
                            child_node = self._build_recursive(grandchild, clause_node)
                            if child_node:
                                clause_node.add_child(child_node)
                        elif isinstance(grandchild, list):
                            for item in grandchild:
                                if isinstance(expr, exp.Table) and isinstance(item, exp.Identifier):
                                    continue  # Skip Identifier under Table
                                if isinstance(item, exp.Expression):
                                    child_node = self._build_recursive(item, clause_node)
                                    if child_node:
                                        clause_node.add_child(child_node)
        if parent is None:
            return root_node

    def _build_recursive(self, sqlglot_node: Expression, parent_node: TreeNode = None) -> TreeNode | None:
        if isinstance(sqlglot_node, exp.Identifier) and parent_node.name == "ColumnRef":
            return None

        if isinstance(sqlglot_node, exp.Identifier) and parent_node.name == "TableRef":
            return None
        
        if isinstance(sqlglot_node, exp.Identifier) and parent_node.name == "Alias":
            return None

        if isinstance(sqlglot_node, exp.TableAlias) and parent_node.name == "TableRef":
            return None

        if isinstance(sqlglot_node, exp.Paren):
            # return first child of the paren expression to avoid unnecessary nesting
            first_child = next(iter(sqlglot_node.args.values()))
            return self._build_recursive(first_child, parent_node)

        if isinstance(sqlglot_node, exp.Select):
            current = self.build_tree(sqlglot_node, parent_node)
        else:
            current = self.create_node(sqlglot_node, parent_node)
            for _, child in sqlglot_node.args.items():
                if isinstance(child, exp.Expression):
                    if isinstance(sqlglot_node, exp.Table) and isinstance(child, exp.Identifier):
                        continue
                    child_node = self._build_recursive(child, current)
                    if child_node:
                        current.add_child(child_node)
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(sqlglot_node, exp.Table) and isinstance(item, exp.Identifier):
                            continue
                        if isinstance(item, exp.Expression):
                            child_node = self._build_recursive(item, current)
                            if child_node:
                                current.add_child(child_node)

        return current

    def walk(self):
        """Yields all nodes in the custom tree in pre-order traversal."""
        if not self.root:
            return
        
        nodes_to_visit = [self.root]
        while nodes_to_visit:
            current = nodes_to_visit.pop(0)
            yield current
            nodes_to_visit.extend(current.children)

    def get_node_by_id(self, node_id: str) -> TreeNode | None:
        """Gets a TreeNode from the custom tree by its ID."""
        return self.id_to_node_map.get(node_id)

    def get_parent(self, node_id: str) -> TreeNode | None:
        """Gets the immediate parent of a node."""
        node = self.get_node_by_id(node_id)
        return node.parent if node else None

    def get_parent_clause(self, node_id: str) -> TreeNode | None:
        """Finds the first ancestor of a node that is a 'Clause'."""
        current_node = self.get_node_by_id(node_id)
        while current_node:
            if current_node.kind == "Clause":
                return current_node
            current_node = current_node.parent
        return None

    def get_parent_statement(self, node_id: str) -> TreeNode | None:
        """Finds the first ancestor of a node that is a 'Statement'."""
        current_node = self.get_node_by_id(node_id)
        while current_node:
            if current_node.kind == "Statement" or current_node.name == "Subquery":
                return current_node
            current_node = current_node.parent
        return None

    def get_tables_in_statement(self, stmt_node: TreeNode) -> list[TreeNode]:
        """Finds all TableRef TreeNodes within a given Statement TreeNode."""
        tables = []
        if stmt_node.kind != "Statement":
            return []
        
        from_clause = next((c for c in stmt_node.children if c.name == 'From'), None)
        if from_clause:
            for node in from_clause.children:
                if node.name == "TableRef":
                    tables.append(node)
        return tables
    
    def find_immediate_subqueries(self, node: 'TreeNode') -> list['TreeNode']:
        """
        Finds all Subquery nodes that are descendants of the given `node`
        but not nested inside another subquery within that same scope.
        """
        subqueries = []
        
        def _walker(current_node, is_top_level=True):
            if not is_top_level and current_node.name == 'Subquery':
                subqueries.append(current_node)
                return # Stop descending further

            for child in current_node.children:
                _walker(child, is_top_level=False)

        _walker(node)
        return subqueries

    def get_tables_in_from_clause(self, stmt_node: 'TreeNode') -> list['TreeNode']:
        """
        Finds the FROM clause of a statement and returns all TableRef nodes within it.
        This needs to handle simple FROMs and complex JOINs.
        """
        # This is a simplified example. A full implementation would need
        # to correctly navigate different statement types (SELECT, UPDATE, etc.)
        # and their structures to find the FROM/JOIN clauses.
        from_or_join_nodes = [
            n for n in stmt_node.walk() 
            if n.name in ('FromClause', 'JoinClause', 'UpdateClause', 'InsertClause', 'DeleteClause') and self.get_parent_statement(n.id) == stmt_node
        ]
        # print(f"Found {len(from_or_join_nodes)} FROM/Join nodes in statement {stmt_node.id}.")
        
        table_refs = []
        for node in from_or_join_nodes:
            for child in node.walk():
                if child.name == 'TableRef' and child not in table_refs:
                    table_refs.append(child)
        return table_refs


    def get_select_list(self, stmt_node: 'TreeNode') -> list['TreeNode']:
        """
        Finds the SELECT clause of a statement and returns a list of its
        projection items (e.g., the nodes for columns, aliases, literals).
        """
        # Simplified example:
        for child in stmt_node.children:
            if child.name == 'Select':
                # The children of the SelectNode are the projection items.
                return child.children
        return []

    def pretty_print(self):
        """Prints a visual, indented tree of the custom AST."""
        if not self.root:
            print("-- Tree is not built --")
            return
        
        def print_recursive(node, prefix="", is_last=True):
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}{node}")
            
            for i, child in enumerate(node.children):
                new_prefix = prefix + ("    " if is_last else "│   ")
                print_recursive(child, new_prefix, is_last=(i == len(child.parent.children) - 1))
        
        print_recursive(self.root)
