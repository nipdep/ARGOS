try:
    from experiment.utils import load_local_module
except ModuleNotFoundError:
    from utils import load_local_module


def get_base_module():
    return load_local_module("din_sql_bird_base", "DIN-SQL_BIRD.py")


def get_dbms_module():
    return load_local_module("din_sql_bird_dbms", "DIN-SQL_BIRD_dbms_access_control.py")
