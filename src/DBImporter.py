import sqlalchemy

class DBImporter:
    def __init__(self, db_url):
        self.db_url = db_url
        self.engine = sqlalchemy.create_engine(self.db_url)
        self.inspector = sqlalchemy.inspect(self.engine)

    def crawl(self):
        """
        Crawl the database schema and extract table and column information.
        """
        # get database name
        db_name = self.inspector.default_schema
        # get tables in the selected database
        tables = self.inspector.get_table_names()
        # get column data dictionary of the selected database
        column_dict = {}
        for table_name in tables:
            primary_key = self.inspector.get_pk_constraint(table_name)['constrained_columns']
            for pk in primary_key:
                column_dict[pk] = {
                    'table': table_name,
                    'type': 'PRIMARY KEY',
                    'nullable': False
                }

            foreign_keys = self.inspector.get_foreign_keys(table_name)
            for fk in foreign_keys:
                for col in fk['constrained_columns']:
                    column_dict[col] = {
                        'table': table_name,
                        'type': 'FOREIGN KEY',
                        'nullable': False
                    }
            
            for column_name in self.inspector.get_columns(table_name):
                if column_name['name'] not in primary_key or column_name['name'] not in foreign_keys:
                    column_dict[column_name['name']] = {
                        'table': table_name,
                        'type': 'COMMON',
                        'nullable': False
                    }

        return db_name, tables, column_dict
    
