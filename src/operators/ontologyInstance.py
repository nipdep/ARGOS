import uuid
import copy
import os
from typing import List, Dict, Any
from sqlglot import parse_one, exp
from sqlglot.expressions import Expression, Identifier, Column

from owlready2 import get_ontology, sync_reasoner_pellet, destroy_entity, sync_reasoner, World

from src.data.ASTTree import TreeNode
from src.operators.astObject import SqlglotOperator
from src.operators.astTree import ASTTreeOperator


class OntologyOperator:
    """Orchestrates the resolution and instantiation of an ontology from an AST tree."""

    def __init__(self, onto_path: str, populated_path: str=None):
        # self.tree_op = tree_operator
        # load ontology as TBOX, ABOX pattern
        self.world = World()
        self.tbox_onto = self.world.get_ontology(onto_path).load()
        self.onto = self.world.get_ontology(onto_path)
        self.onto.imported_ontologies.append(self.tbox_onto)
        if populated_path and os.path.exists(populated_path):
            # append the populated ontology if exists
            self.abox_onto = self.world.get_ontology(populated_path).load()
            self.onto.imported_ontologies.append(self.abox_onto)

        self.class_lookup = {cls.name: cls for cls in self.onto.classes()}
        self.created_instances = []
        self.table_ref_instances = []
        self.column_ref_instances = []

        self._create_object_mapper()
        # self._build_policy_params()

    def _create_object_mapper(self):
        """
        Creates a mapping of table and column names to their corresponding
        """
        self.table_entities = {
            t.TableName[0]: t.name for t in self.get_individuals_by_class("Table")}
        self.column_lookup = {(tbl.TableName[0], c.ColumnName[0]): c.name for c in self.get_individuals_by_class(
            "Column") for tbl in c.columnOfTable}
        
        print(f"table entities: {self.table_entities}")
        print(f"column lookup: {self.column_lookup}")

    def get_table_ref_instances(self):
        """
        Returns the list of instantiated TableRef objects.
        """
        table_ref_dict = {}
        for inst in self.table_ref_instances:
            status = getattr(inst, 'hasStatus', [])
            instance_dict = {}
            if not status:
                instance_dict['Status'] = "Aligned"
            else:
                instance_dict['Status'] = status[0].name
                instance_dict['Policy'] = getattr(
                    inst, 'relatedPolicy', [None])[0].name
                instance_dict['Rule'] = getattr(
                    inst, 'inferredRule', [None])[0].name

            table_ref_dict[inst.name] = instance_dict
            # table_ref_dict[inst.name]
        # return {inst.name: "Aligned" if getattr(inst, 'hasStatus', []) == [] else mapper.get(str(getattr(inst, 'hasStatus', [])[0])) for inst in self.table_ref_instances }
        return table_ref_dict

    def get_column_ref_instances(self):
        """
        Returns the list of instantiated ColumnRef objects.
        """
        column_ref_dict = {}
        for inst in self.column_ref_instances:
            status = getattr(inst, 'hasStatus', [])
            instance_dict = {}
            if not status:
                instance_dict['Status'] = "Aligned"
            else:
                instance_dict['Status'] = status[0].name
                instance_dict['Policy'] = getattr(
                    inst, 'relatedPolicy', [None])[0].name
                instance_dict['Rule'] = getattr(
                    inst, 'inferredRule', [None])[0].name

            column_ref_dict[inst.name] = instance_dict
        # return {inst.name: "Aligned" if getattr(inst, 'hasStatus', []) == [] else "Violated" for inst in self.column_ref_instances }
        return column_ref_dict

    def get_statistics(self) -> dict:
        """
        Returns a dictionary with statistics about the ontology instances.
        """
        return {
            "total_instances": len(self.created_instances),
            "table_refs": len(self.table_ref_instances),
            "column_refs": len(self.column_ref_instances),
            "table_entities": len(self.table_entities),
            "column_lookup": len(self.column_lookup)
        }

    def get_individuals_by_class(self, class_name: str) -> list:
        """
        Gets all individuals that are instances of a given class name.
        """
        # Find the actual class object from the lookup map created in __init__
        target_class = self.class_lookup.get(class_name)

        if not target_class:
            print(f"[Warning] Class '{class_name}' not found in the ontology.")
            return []

        # The onto.search() method efficiently finds all individuals of the given type.
        return self.onto.search(type=target_class)

    def get_node_attribute(self, node_id: str, attr_name: str) -> str | None:
        """
        Retrieves a specific attribute from an ontology instance (node).
        """
        try:
            node = self.onto[node_id]
            # print all attributes of the node
            # for attr in dir(node):
            #     if not attr.startswith('_') and not callable(getattr(node, attr)):
            #         print(f"node_id: {node_id} | Attribute: {attr} | Value: {getattr(node, attr)}")

            value = getattr(node, attr_name, None)

            return str(value) if value is not None else None
        except (KeyError, AttributeError):
            return None
    
    def get_instance_by_id(self, id: str):
        """
        Retrieves an ontology instance by its ID.
        """
        return self.onto.get(id)

    def _build_policy_params(self):
        """
        Builds the parameters for the policy creation.
        """
        self.policy_params = {
            "Aligned": self.onto.get("Aligned"),
            "ColumnLevel": self.onto.get("ColumnLevel"),
            "Conditional": self.onto.get("Conditional"),
            "DeleteActionScope": self.onto.get("DeleteActionScope"),
            "InsertActionScope": self.onto.get("InsertActionScope"),
            "ModifyAction": self.onto.get("ModifyAction"),
            "Permitted": self.onto.get("Permitted"),
            "ProcessActionScope": self.onto.get("ProcessActionScope"),
            "Prohibited": self.onto.get("Prohibited"),
            "ReadAction": self.onto.get("ReadAction"),
            "RowLevel": self.onto.get("RowLevel"),
            "RowTag": self.onto.get("RowTag"),
            "TableLevel": self.onto.get("TableLevel"),
            "UpdateActionScope": self.onto.get("UpdateActionScope"),
            "ViewActionScope": self.onto.get("ViewActionScope"),
            "Violated": self.onto.get("Violated")
        }

    def create_schema(self, db_name: str, table_list: List[str], column_dict: Dict[str, Dict[str, Any]]) -> None:
        """
        Create the schema instance of the importing database
        """
        table_instance_map = {}
        with self.onto:
            # create database instance
            db_instance = self.onto.Database(DatabaseName=db_name)

            # create table instance related to the created database instance
            for table_name in table_list:
                table_instance = self.onto.Table(TableName=table_name)
                db_instance.hasTable.append(table_instance)
                table_instance_map[table_name] = table_instance

            # create column instances related to the created table instances
            for col_name, col_info in column_dict.items():
                table_name = col_info['table']
                if table_name in table_list:
                    col_instance = self.onto.Column(ColumnName=col_name)
                    if col_info['type'] == 'PRIMARY KEY':
                        table_instance_map[table_name].primaryKey.append(col_instance)
                    elif col_info['type'] == 'FOREIGN KEY':
                        table_instance_map[table_name].foreignKey.append(col_instance)
                    else:
                        table_instance_map[table_name].hasColumn.append(col_instance)

    def create_policy(self, agent_name: str, grant_type: str, grant_level: str, action: str, action_scope: str, resource: List[str], condition: str=None):
        """
        Creates a policy instance in the ontology.
        """
        with self.onto: 
            policy_instance = self.onto.Policy()
            # add object relation to this instance 
            policy_instance.hasAgent.append(self.policy_params["Agent"])
            policy_instance.hasGrantType.append(self.policy_params["GrantType"])
            policy_instance.hasGrantLevel.append(self.policy_params["GrantLevel"])
            policy_instance.hasAction.append(self.policy_params["Action"])
            policy_instance.hasActionScope.append(self.policy_params["ActionScope"])
            for resource in resource:
                policy_instance.hasResource.append(self.onto.get(resource))
                
            if condition:
                policy_instance.hasCondition.append(condition)

    def close(self):
        """
        Closes the ontology connection and cleans up resources.
        """
        self.onto.destroy()

    def cleanup(self):
        """
        Deletes a list of individuals from the ontology by their IDs.
        """
        self.column_ref_instances = []
        self.table_ref_instances = []
        # print(f"--> Created instances : {[inst.name for inst in self.created_instances]}")

        if not self.created_instances:
            print("--> Cleanup: No instance IDs provided to delete.")
            return

        print(
            f"--> Cleanup: Attempting to delete {len(self.created_instances)} instances...")
        deleted_count = 0

        # Use a 'with' block for modifications, which is good practice
        with self.onto:
            for inst_id in self.created_instances:
                # Find the individual by its name (which is its ID in our case)
                # The '*' is a wildcard for the IRI search
                individual_to_delete = self.onto.search_one(
                    iri=f"*{inst_id.name}")
                # print(f"    - Attempting to delete instance: {inst_id} ({individual_to_delete})")
                if individual_to_delete:
                    try:
                        # print(f"    - Deleting instance: {inst_id} ({individual_to_delete})")
                        # The correct function to remove an individual
                        destroy_entity(individual_to_delete)
                        print(f"    - Deleted instance: {inst_id}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"    - Error deleting instance {inst_id}: {e}")
                else:
                    print(f"    - Instance not found for deletion: {inst_id}")
                # self.created_instances.remove(inst_id)  # Remove from the list after deletion attempt

            print(
                f"--> Cleanup complete. Successfully deleted {deleted_count} instances.")

    def update_lookups(self, with_clause: TreeNode):
        alias = with_clause.alias
        self.table_entities[alias] = "t00x"
        subquery = with_clause
        for node in subquery.walk():
            if node.name == "ColumnRef":
                col_name = node.refcol
                # table_name = node.reftable
                if col_name:
                    # XXX: future issue could be happen here
                    if hasattr(node, "column_reference_id"):
                        self.column_lookup[(alias, col_name)] = node.column_reference_id
                        print(f"Updated column lookup: {alias}.{col_name} -> {node.column_reference_id}")
                    else:
                        self.column_lookup[(alias, col_name)] = "c00x"
            # elif node.name == "TableRef":
            #     # table_name = node.reftable
            #     self.table_entities[alias] = node.table_reference_id
            #     print(f"Updated table entities: {alias} -> {node.table_reference_id}")

    def add_CTE_to_lookup(self):
        # get all with clauses in query 
        with_clauses = self.tree_op.get_with_clauses(self.tree_op.root)
        for with_clause in with_clauses:
            self.update_lookups(with_clause)

    def resolve_wildcard(self):
        """
        Finds and expands all wildcards in the query. This modifies the
        underlying AST and rebuilds the custom tree. Call before instantiation.
        """
        wildcard_nodes = [node for node in self.tree_op.walk() if (
            node.name == "Wildcard") and (node.parent.name != "Function")]
        if not wildcard_nodes:
            return

        # print(f"--> Found {len(wildcard_nodes)} wildcard(s). Expanding...")
        for wc_node in wildcard_nodes:
            parent_node = self.tree_op.get_parent(wc_node.id)
            parent_stmt_treenode = self.tree_op.get_parent_statement(wc_node.id)
            with_clause = self.tree_op.get_with_clauses(parent_stmt_treenode)

            # print(f"Wildcard node id: {wc_node.id} | Parent node: {parent_node.id} | Parent statement: {parent_stmt_treenode.id}")
            wildcard_sqlglot_node = self.tree_op.sql_op.get_node_by_id(
                wc_node.id)
            # print(f"Wildcard SQLGlot node: {wildcard_sqlglot_node}")
            parent_select_sqlglot_node = wildcard_sqlglot_node.find_ancestor(
                exp.Select)

            if not parent_stmt_treenode:
                return
            elif with_clause:
                with_aliases = [cl.alias for cl in with_clause]
                for col in self.column_lookup:
                    # print(f"Expanding wildcard in WITH clause: {with_aliases} for column: {col[1]}")
                    if col[0] in with_aliases:
                        col_name = col[1]
                        new_node_id = "n" + str(uuid.uuid4())[:6]
                        # print(f"Column name : {col_name} | New node ID: {new_node_id}, Table name: {table_name}")
                        new_col_node = self.tree_op.create_column_node(
                            parent_sqlglot_node=parent_select_sqlglot_node,
                            parent_node=parent_node,
                            node_id=new_node_id,
                            refcol=col_name,
                            reftable=col[0]
                        )

                self.tree_op.remove_node_by_id(wc_node.id)
            else:
                # resolve table reference 
                # with alias 
                if hasattr(wc_node, "alias"):
                    # Resolve table reference with alias
                    print(f"Resolving wildcard with alias: {wc_node.alias}")
                    table_alias = wc_node.alias
                    table_ref = self.tree_op.get_table_ref_by_alias(table_alias)
                    if table_ref:
                        table_name = table_ref.reftable
                    else:
                        print(f"[Warning] Table alias '{table_alias}' not found.")
                        return
                else:
                    tables_in_stmt = self.tree_op.get_tables_in_from_clause(
                        parent_stmt_treenode)
                    # print(f" tables_in_stmt: {tables_in_stmt}")
                    table_name = tables_in_stmt[0].reftable
                    if len(tables_in_stmt) != 1:
                        print(
                            f"[Warning] Wildcard expansion only supported for single-table queries. Skipping.")
                        return

                table_ind = self.onto.search_one(TableName=table_name)
                if not table_ind or not hasattr(table_ind, "hasColumn"):
                    print(
                        f"[Warning] Table '{table_name}' not in ontology or has no columns. Cannot expand.")
                    return

                # go through all columns of the table and create ColumnRef nodes
                for col in table_ind.hasColumn:
                    # print(f"Expanding wildcard for column: {col.ColumnName[0]} in table: {table_name}")
                    col_name = col.ColumnName[0]
                    new_node_id = "n" + str(uuid.uuid4())[:6]
                    # print(f"Column name : {col_name} | New node ID: {new_node_id}, Table name: {table_name}")
                    new_col_node = self.tree_op.create_column_node(
                        parent_sqlglot_node=parent_select_sqlglot_node,
                        parent_node=parent_node,
                        node_id=new_node_id,
                        refcol=col_name,
                        reftable=table_name
                    )

                # Remove the wildcard node from the parent statement
                self.tree_op.remove_node_by_id(wc_node.id)

    def resolve_references(self, node: 'TreeNode', parent_context={}) -> dict:
        """
        Recursively resolves table and column references in the AST, mapping them to
        database entity IDs.
        """
        # print(f"Resolving references in node: {node.id} | {node.name}")
        extended_context = {}

        in_context_sources = {}

        with_clauses_in_scope = self.tree_op.get_with_clauses(node)
        for with_clause in with_clauses_in_scope:
            # update directly table_entities and column_lookup dictionaries with the with clause values 
            print(f"Resolving WITH clause: {with_clause}")
            # break
            self.resolve_references(with_clause)
            # update lookups with the with clause values
            self.update_lookups(with_clause)

            # print(f"Resolving WITH clause: {with_clause.id}")
            # with_alias = with_clause.alias

            # exposed_schema = self.resolve_references(
            #     with_clause, in_context_sources)
            # in_context_sources.update(exposed_schema)
            # in_context_sources[with_alias] = {"type": "Synthetic"}

            # in_context_sources[with_clause.alias] = { "type": "with_clause", "schema": exposed_schema }
            # parent = self.tree_op.get_parent(with_clause.id)
            # print(f" exposed_schema: {exposed_schema} | parent: {with_alias}")
            # if parent and parent.name == "Alias":
                # alias = parent.alias
            # for sq_id, sq_context in exposed_schema.items():
                # context_sources[with_alias] = { "type": "base_table", "id": sq_id, "name": sq_context['name'] }
                # extended_context[with_clause.id] = {"alias": with_alias, "schema": exposed_schema}
        # print(f" context_sources after WITH clauses: subquery_context : {extended_context} | in_context_sources: {in_context_sources}")


        from_clause_tables = self.tree_op.get_tables_in_from_clause(node)
        # print(f" node.id: {node.id} | from_clause_tables: {from_clause_tables}")
        # print(f"Found {len(from_clause_tables)} table(s) in FROM clause.")
        output_schema = {}
        for table_node in from_clause_tables:
            table_name = table_node.reftable
            if not in_context_sources.get(table_name):
                in_context_sources[table_name] = {
                    "type": "base_table", "name": table_name}
                output_schema[table_name] = {
                    "type": "base_table", "name": table_name}
                
                alias = getattr(table_node, 'refalias', None)
                if alias:
                    in_context_sources[alias] = {
                        "type": "base_table", "name": table_name}
                    output_schema[alias] = {
                        "type": "base_table", "name": table_name}

        print(f"Initial in-context sources: {in_context_sources}")

        # first we need to resolve custom table created from WITH clauses
        
        # --- First Walk (unchanged) ---
        # then we need to resolve subqueries in the current node
        subqueries_in_scope = self.tree_op.find_immediate_subqueries(node)
        for subquery_node in subqueries_in_scope:
            exposed_schema = self.resolve_references(
                subquery_node, in_context_sources)
            # XXX: probably we need to re-attach this context
            # parent = self.tree_op.get_parent(subquery_node.id)

            # if parent and parent.name == "Alias":
            #     subquery_contexts[subquery_node.id] = {
            #         "alias": parent.alias, "schema": exposed_schema}
        # print(f"Subquery contexts after first walk: {subquery_contexts}")
        
        # --- Finalize In-Context References (unchanged) ---
        # add parent context sources to the in_context_sources
        # in_context_sources = copy.deepcopy(context_sources)
        for alias, source in parent_context.items():
            if alias not in in_context_sources:
                in_context_sources[alias] = source

        # for sq_id, sq_context in subquery_contexts.items():
        #     alias = sq_context['alias']
        #     in_context_sources[alias] = {
        #         "type": "subquery", "id": sq_id, "schema": sq_context['schema']}
            
        # print(f" full context : {in_context_sources}")
        # print(f"Resolved {len(in_context_sources)} in-context sources for node {node.id} | context : {in_context_sources}.")
        # --- Second Walk: UPDATED to use new resolver methods ---
        # print(f"Resolving references in node {node.id} | in_context_sources: {in_context_sources}")
        
        for current_node in self._walk_and_skip_subqueries(node):
            if current_node.name == 'ColumnRef':
                col_name, col_id, table_id = self._resolve_single_column(current_node, in_context_sources)
                output_schema[col_name] = {"type": "extended_column", "col_id": col_id, "table_id": table_id}

            elif current_node.name == 'TableRef':
                # Call the new dedicated table resolver
                table_name, table_id = self._resolve_single_table(current_node, in_context_sources)
                output_schema[table_name] = {"type": "extended_table", "id": table_id}

        # --- Prepare and Return Exposed Schema (unchanged) ---
        # output_schema = {"columns": {}}
        # select_list_nodes = self.tree_op.get_select_list(node)
        # print(f"Select list nodes for node {node.id}: {select_list_nodes}")
        # for item_node in select_list_nodes:
        #     output_name = None
        #     resolved_source_info = "expression"
        #     if hasattr(item_node, 'resolved_source'):
        #         resolved_source_info = item_node.resolved_source
        #         output_name = item_node.refcol
        #     if item_node.name == 'Alias':
        #         output_name = item_node.alias
        #         column_expr_node = item_node.children[0]
        #         if hasattr(column_expr_node, 'resolved_source'):
        #             resolved_source_info = column_expr_node.resolved_source
        #     if output_name:
        #         output_schema['columns'][output_name] = resolved_source_info
        # print(f"Resolved output schema for node {node.id}: {output_schema}")
        return output_schema

    # --- NEW: Dedicated method to resolve table references ---
    def _resolve_single_table(self, table_node: 'TreeNode', context_sources: dict) -> str | None:
        """Helper to resolve a single table reference node to its database entity ID."""
        alias = getattr(table_node, 'refalias', table_node.reftable)
        table_id = None
        # print(f"Resolving table '{table_node}' | context_sources: {context_sources}")
        if alias in context_sources:
            source_info = context_sources[alias]
            table_node.resolved_info = source_info

            # Only base tables have a direct ID in the database ontology.
            if source_info['type'] == 'base_table':
                real_table_name = source_info['name']

                # Use the mapper to get the table's entity ID.
                table_id = self.table_entities.get(real_table_name)

                if table_id:
                    # Attach the resolved database ID to the node.
                    table_node.table_reference_id = table_id
                    # self.table_ref_instances.append(table_node)
                else:
                    raise ValueError(
                        f"Table '{real_table_name}' found in query but not in database entities.")
            elif source_info['type'] == "extended_table":
                table_node.reference_id = source_info['id']
            elif source_info['type'] == "synthetic_table":
                table_node.name = "Alias"
        else:
            table_id = self.table_entities.get(alias)
            table_node.table_reference_id = table_id

        return alias, table_id

    # --- UPDATED: Method to resolve column references ---
    def _resolve_single_column(self, col_node: 'TreeNode', context_sources: dict):
        """Helper to resolve a single column reference node to its database entity ID."""
        col_name = col_node.refcol
        table_alias_in_query = col_node.reftable

        print(f"Resolving column '{col_name}' in context of table alias '{table_alias_in_query}' | context_sources: {context_sources}")
        column_id = None
        table_id = None
        candidate_sources = []
        if table_alias_in_query:
            if table_alias_in_query in context_sources:
                source = context_sources[table_alias_in_query]
                # print(f"Found table alias '{table_alias_in_query}' in context sources: {source}")
                if self._column_exists_in_source(col_name, source):
                    candidate_sources.append(source)
            else:
                raise NameError(
                    f"Table or alias '{table_alias_in_query}' not found.")
        else:
            if col_name in context_sources:
                print(f" node: {col_node.id} checking in context sources for column: {col_name}")
                source = context_sources[col_name]
                if source['type'] == "extended_column":
                    col_node.column_reference_id = source['col_id']
                    col_node.table_reference_id = source['table_id']

                    return col_name, col_node.column_reference_id, col_node.table_reference_id
            else:
                # get the base table from the context sources
                for source_alias, source in context_sources.items():
                    # Check if the column exists in this source
                    if source['type'] == 'base_table':
                        # If the column exists in this base table, add it to candidates
                        if self._column_exists_in_source(col_name, source):
                            candidate_sources.append(source)
                    # if self._column_exists_in_source(col_name, source):
                    #     candidate_sources.append(source)
                    # elif source['type'] == "extended_column":
                    #     col_node.reference_id = source['id']
                    #     return col_name, col_node.reference_id

        if candidate_sources:  # len(candidate_sources) == 1:
            source = candidate_sources[0]
            source_alias = next(
                key for key, val in context_sources.items() if val == source)

            col_node.reftable = source_alias

            # If the source is a base table, perform the ID lookup.
            if source['type'] == 'base_table':
                real_table_name = source['name']
                table_id = self.table_entities.get(real_table_name)

                if table_id:
                    # Attach the resolved database ID to the node.
                    col_node.table_reference_id = table_id

                col_node.resolved_source = f"{real_table_name}.{col_name}"

                # Use the mapper to get the column's entity ID.
                column_id = self.column_lookup.get((real_table_name, col_name))
                # print(f"Column '{col_name}' in table '{real_table_name}' resolved to ID: {column_id}")
                if column_id:
                    # Attach the resolved database ID to the node.
                    col_node.column_reference_id = column_id
                    # self.column_ref_instances.append(col_node)

                else:
                    raise ValueError(
                        f"Column '{col_name}' in table '{real_table_name}' not in database entities.")
            else:
                # Source is a subquery; it has no direct database ID.
                subquery_alias = source.get('alias', 'subquery')
                print(
                    f"[Warning] Column '{col_name}' in subquery '{subquery_alias}' has no database ID.")
                col_node.resolved_source = f"{subquery_alias}.{col_name}"
                # self.column_ref_instances.append(col_node)

        # elif len(candidate_sources) > 1:
        #     raise ValueError(
        #         f"Ambiguous column reference: '{col_name}' exists in multiple tables.")
        else:
            raise ValueError(
                f"Could not resolve column reference: '{col_name}'.")
        
        return col_name, column_id, table_id

    def _column_exists_in_source(self, column_name: str, source_info: dict) -> bool:
        """Checks if a column exists in a given data source (table or subquery)."""
        # print(f"Checking if column '{column_name}' exists in source: {source_info}")
        if source_info['type'] == 'base_table':
            table_name = source_info['name']
            # This is where you would query your ontology.
            # For example, check if `column_name` is a data property of the
            # class corresponding to `table_name`.
            # return self.ontology.is_property_of(column_name, table_name)
            # Placeholder logic:
            # table_class = self.class_lookup.get(table_name)
            return self.column_lookup.get((table_name, column_name), None) is not None

        elif source_info['type'] == 'subquery':
            # Check against the exposed schema of the subquery.
            # XXX: check
            return column_name in source_info['schema']['columns']

        return False

    def _walk_and_skip_subqueries(self, node: 'TreeNode'):
        """
        A generator that yields a node and its descendants. It stops 
        descending into any child that is a 'Subquery'.
        """
        yield node  # Yield the current node.

        # Iterate through children and decide whether to recurse.
        for child in node.children:
            # If the child is a subquery, we yield it but do not walk its children.
            # This is because the main `resolve_references` function has already processed
            # it in the first walk.
            if child.name == 'Subquery' or child.name == 'WithClause':
                yield child
            else:
                # If the child is not a subquery, recurse into its subtree.
                yield from self._walk_and_skip_subqueries(child)

    def instantiate_ontology(self, tree_operator: ASTTreeOperator, agent_id: str):
        """Walks the tree and creates ontology individuals."""
        print("--> Instantiating ontology individuals...")
        self.created_instances = []
        self.table_ref_instances = []
        self.column_ref_instances = []
        self.tree_op = tree_operator
        with self.onto:
            self.add_CTE_to_lookup()
            self.resolve_wildcard()
            print(self.tree_op.sql_op.ast)
            self.resolve_references(self.tree_op.root)

            self._instantiate_recursive(self.tree_op.root, agent_id)
        print(
            f"    - Instantiated {len(self.table_ref_instances)} TableRefs and {len(self.column_ref_instances)} ColumnRefs.")

    def _instantiate_recursive(self, node: TreeNode, agent_id: str, parent_instance=None, parent=None, clause_instance=None):
        node_class = self.class_lookup.get(node.name)
        # print(f"Instantiating node: {node.id} | Class: {node_class} | Parent: {parent_instance if parent_instance else None}")
        if not node_class:
            # Keep traversing through unmapped wrappers (e.g. Not/Is/Null) so
            # nested ColumnRef/TableRef nodes still get instantiated.
            for child in node.children:
                self._instantiate_recursive(
                    child,
                    agent_id,
                    parent_instance=parent_instance,
                    parent=parent,
                    clause_instance=clause_instance,
                )
            return

        inst = node_class(node.id)
        self.created_instances.append(inst)
        if parent_instance:
            if (parent.kind == "Statement") and (node.kind == "Clause"):
                parent_instance.hasClause.append(inst)
            elif (parent.kind == "Clause") and (node.kind == "Expression"):
                parent_instance.hasExpression.append(inst)
            else:
                parent_instance.immediateChildNode.append(inst)

        if clause_instance and node.kind != "Clause":
            inst.ofClauseNode.append(clause_instance)

        if node.kind == "Statement":
            agent_ref = self.onto.search_one(iri=f"*{agent_id}")
            if agent_ref:
                inst.executedBy.append(agent_ref)

        if node.kind == "Operator":
            pass
            # inst.value = 

        if hasattr(node, "table_reference_id") and node.table_reference_id:
            table_obj = self.onto.search_one(iri=f"*{node.table_reference_id}")
            if table_obj:
                if node.name == "TableRef":
                    inst.referencesToTable.append(table_obj)
                    self.table_ref_instances.append(inst)
                elif node.name == "ColumnRef":
                    inst.referencesToTable.append(table_obj)
                    if node.column_reference_id:
                        column_obj = self.onto.search_one(
                            iri=f"*{node.column_reference_id}")
                        if column_obj:
                            inst.referencesToColumn.append(column_obj)
                            # print(f"ColumnRef {node.id} references column {node.column_reference_id} in table {node.table_reference_id}")
                    # print(f"ColumnRef {node.id} references column {node.reference_id}")
                    self.column_ref_instances.append(inst)

        for child in node.children:
            if node.kind == "Clause":
                clause_instance = inst
            self._instantiate_recursive(
                child, agent_id, parent_instance=inst, parent=node, clause_instance=clause_instance)

    def reason_and_save(self, output_path: str, save=False):
        """Runs the reasoner and saves the final ontology."""
        print("--> Running reasoner and saving ontology...")
        reasoner_targets = [self.onto]
        reasoner_debug = 1 if os.getenv("OWLREADY_REASONER_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"} else 0
        with self.onto:
            # sync_reasoner(infer_property_values=True)
            # IMPORTANT: pass ontology targets explicitly; otherwise Owlready2
            # reasons over default_world and skips this operator's custom World().
            sync_reasoner_pellet(
                reasoner_targets,
                infer_property_values=True,
                debug=reasoner_debug,
            )

        with self.onto:  # XXX: this proves that pellet reasoner runs SWRL rule chaining sequentially.
            sync_reasoner_pellet(
                reasoner_targets,
                infer_property_values=True,
                debug=reasoner_debug,
            )

        # with self.onto: # XXX: this proves that pallet reasoner run reasoning rule sequentially.
        #     sync_reasoner_pellet(infer_property_values = True)

        if save:
            self.onto.save(file=output_path, format="rdfxml")
            print(f"    - Ontology saved to {output_path}")
