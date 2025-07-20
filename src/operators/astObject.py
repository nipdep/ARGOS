import uuid
from sqlglot import parse_one, exp
from sqlglot.expressions import Expression


class SqlglotOperator:
    """
    A dedicated class to encapsulate and operate on a sqlglot AST object.

    This operator automatically parses a SQL string, decorates the resulting
    AST with unique IDs for every node, and provides a suite of methods
    for easy and robust inspection and manipulation of the query structure.
    """

    def __init__(self, sql: str):
        """
        Initializes the operator by parsing the SQL and preparing the AST.
        """
        try:
            self.ast: Expression = parse_one(sql)
            self._decorate_ast_with_ids()
            self._map_ids_to_nodes()
        except Exception as e:
            print(f"Error initializing SqlglotOperator: {e}")
            self.ast = None
            self.id_to_node_map = {}

    def _decorate_ast_with_ids(self):
        """
        Private helper to traverse the AST and assign a unique ID to a 'meta'
        attribute on each node. This is fundamental for all ID-based operations.
        """
        if not self.ast:
            return
        for node in self.ast.walk():
            if not hasattr(node, 'meta'):
                node.meta = {}
            node.meta['id'] = "n"+str(uuid.uuid4())[:6]

    def _map_ids_to_nodes(self):
        """
        Private helper to create a dictionary mapping unique IDs to their
        corresponding nodes for efficient lookups.
        """
        if not self.ast:
            self.id_to_node_map = {}
            return
        self.id_to_node_map = {
            node.meta['id']: node for node in self.ast.walk() if hasattr(node, 'meta')
        }
        # print(f"Mapped {len(self.id_to_node_map)} nodes to their IDs.\n Mapped node Ids: {list(self.id_to_node_map.keys())}...")  # Print first 10 IDs for brevity

    def to_sql(self, pretty: bool = True) -> str:
        """
        Generates the SQL string from the current state of the AST.
        """
        if not self.ast:
            return None
        return self.ast.sql(pretty=pretty)

    def get_node_by_id(self, node_id: str) -> Expression | None:
        """
        Retrieves a single node from the AST by its unique ID.
        """
        # print(f"Looking for node with ID: {node_id}")
        # print(f"Total nodes in AST: {len(self.id_to_node_map)} | {self.id_to_node_map.keys()}")
        return self.id_to_node_map.get(node_id)

    def get_nodes_by_type(self, node_type: type) -> list[Expression]:
        """
        Finds all nodes in the AST that are of a specific type.
        """
        if not self.ast:
            return []
        return self.ast.find_all(node_type)

    def remove_node_by_id(self, target_id: str):
        """
        Removes a single node from the AST based on its unique ID.
        The internal AST is updated with the result.
        """
        if not self.ast or not target_id:
            return

        def transformer(node):
            if hasattr(node, 'meta') and node.meta.get('id') == target_id:
                return None  # Returning None deletes the node
            return node

        # transform() returns a new AST object. We must update our instance.
        new_ast = self.ast.transform(transformer)
        self.ast = new_ast
        # After modification, the ID map needs to be rebuilt.
        self.id_to_node_map.pop(target_id, None)  # Remove the ID from the map
        
    def add_node(self, parent_id: str, new_node: Expression, arg_name: str):
        """
        Adds a new node as a child of an existing node.
        """
        parent_node = self.get_node_by_id(parent_id)
        print(f"Adding new node with ID '{new_node}' under parent node '{parent_node}'")
        if not parent_node:
            print(f"[Error] Parent node with ID '{parent_id}' not found.")
            return

        # Decorate the new node and its children with IDs before adding
        # self._decorate_ast_with_ids_on_subtree(new_node)

        # Handle list arguments (like 'expressions' or 'joins')
        if isinstance(getattr(parent_node, arg_name, None), list):
            # FIX: Get the list attribute dynamically using arg_name
            list_arg = getattr(parent_node, arg_name)
            # FIX: Append to the list we just retrieved
            list_arg.append(new_node)
        # Handle single arguments (like 'where' or 'limit')
        else:
            parent_node.set(arg_name, new_node)
        
        # Rebuild the map since we've added new nodes
        # self._map_ids_to_nodes()

    def replace_node(self, target_id: str, new_node: Expression):
        """
        Replaces a target node with a new node.
        """
        if not self.ast or not target_id:
            return
            
        self._decorate_ast_with_ids_on_subtree(new_node)

        def transformer(node):
            if hasattr(node, 'meta') and node.meta.get('id') == target_id:
                return new_node
            return node
            
        self.ast = self.ast.transform(transformer)
        self._map_ids_to_nodes()

    def add_where_condition(self, condition_sql: str):
        """
        Adds a new condition to the main WHERE clause of the query from a string.
        """
        if not self.ast:
            print("[Error] AST is not initialized.")
            return

        # The .where() helper on the main query expression is the easiest way.
        # It handles both creating a new WHERE and AND-ing to an existing one.
        # copy=False ensures the modification happens in-place.
        self.ast.where(condition_sql, copy=False)

        # After modification, the AST structure has changed, so we need to
        # re-decorate the new parts and rebuild the ID map.
        self._decorate_ast_with_ids_on_subtree(self.ast)
        # self._map_ids_to_nodes()

    def _decorate_ast_with_ids_on_subtree(self, ast_node: Expression):
        """Helper to add IDs to a new subtree before it's inserted."""
        for node in ast_node.walk():
            if not hasattr(node, 'meta'):
                node.meta = {}
            if 'id' not in node.meta:
                 node.meta['id'] = str(uuid.uuid4())

    def pretty_print(self):
        """Prints a visual, indented tree of the current AST with node IDs."""
        if not self.ast:
            print("-- Invalid AST")
            return
            
        for node in self.ast.walk():
            indent = "  " * node.depth
            short_id = node.meta.get('id', 'N/A')[:8]
            node_name = node.__class__.__name__
            
            # Add extra detail for common nodes
            if isinstance(node, exp.Identifier):
                node_name += f" ({node.this})"
                continue
            elif isinstance(node, exp.Column):
                 node_name += f" ({node.sql()})"

            print(f"{indent}├── {node_name}  [ID: {short_id}]")