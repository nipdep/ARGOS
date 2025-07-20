
from sqlglot import exp

class TreeNode:
    """ 
    TreeNode represents a node in the SQL AST tree. It can be a statement, clause, or expression.
    """
    def __init__(self, sqlglot_node, kind, name, parent=None):
        self.remove = False
        self.parent = parent
        self.children = []
        self.sqlglot_node = sqlglot_node
        self.kind = kind
        self.name = name

        if self.kind == "Statement":
            self.id = "s" + sqlglot_node.meta['id'][1:]
        else:
            self.id = sqlglot_node.meta['id']
        # sqlglot_node.meta['id'] = self.id

        if self.name == "ColumnRef":
            self.refcol = None
            self.reftable = None
            # self.reference_id = None
            parts = sqlglot_node.parts
            if len(parts) == 2:
                self.reftable = parts[0].this
                self.refcol = parts[1].this
            elif len(parts) == 1:
                self.refcol = parts[0].this

        if self.name == "TableRef":
            self.reftable = sqlglot_node.name
            if sqlglot_node.alias:
                self.refalias = sqlglot_node.alias
            # self.reference_id = None

        if self.name == "Alias":
            alias_expr = sqlglot_node.args.get("alias")
            if isinstance(alias_expr, exp.Identifier):
                self.alias = alias_expr.name

        if self.name == "CTE":
            alias_expr = sqlglot_node.args.get("alias")
            if isinstance(alias_expr, exp.TableAlias):
                self.alias = alias_expr.this.name

        if self.name == "Literal":
            self.value = sqlglot_node.this
        
        if self.name == "Operator":
            if (type(sqlglot_node).__name__ in ["And", "Or"]):
                self.logics = True 
            else:
                self.logics = False    

    def add_child(self, child):
        self.children.append(child)

    def remove_child(self, child):
        if child in self.children:
            self.children.remove(child)
        else:
            raise ValueError(f"Child {child} not found in children of {self}")

    def __bool__(self):
        # A custom truthiness check: Node is True if it has a non-None value
        return self.id is not None

    def __eq__(self, other):
            if not isinstance(other, TreeNode):
                return NotImplemented  # Or raise TypeError
            return self.id == other.id #and self.kind == other.kind and self.name == other.name

    def __repr__(self):
        suffix = f" ({self.name})"
        if self.name == "ColumnRef":
            suffix += f" [reftable={self.reftable}, refcol={self.refcol}]"
        if self.name == "TableRef":
            suffix += f" [reftable={self.reftable}]"
        if self.name == "Literal" and hasattr(self, "value"):
            suffix += f" [value={self.value}]"

        if hasattr(self, "alias"):
            suffix += f" [alias={self.alias}]"
        if hasattr(self, "table_reference_id"):
            suffix += f" [table_reference_id={self.table_reference_id}]"
        if hasattr(self, "column_reference_id"):
            suffix += f" [column_reference_id={self.column_reference_id}]"
        if hasattr(self, "remove"):
            suffix += f" [remove={self.remove}]"
        if hasattr(self, "refalias"):
            suffix += f" [refalias={self.refalias}]"
        if hasattr(self, "logics"):
            suffix += f" [logics={self.logics}]"
        return f"<({self.id}) {self.kind} | {suffix}>"

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()

    def print_tree(self, level=0):
        print("  " * level + repr(self))
        for child in self.children:
            child.print_tree(level + 1)
