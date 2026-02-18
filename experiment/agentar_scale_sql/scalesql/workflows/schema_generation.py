import argparse
import yaml
import logging
from typing import Any, Dict

from scalesql.modules.light_schema import LightSchema
from scalesql.utils import (
    read_json,
    save_or_append_json,
    read_env,
    setup_logging,
    get_cursor_from_path
)

setup_logging()


class SchemaGeneration:
    @staticmethod
    def save_json(
            database_schemas: Dict[str, Any], save_path: str, overwrite: bool = True
    ) -> None:
        save_or_append_json(
            data=database_schemas, filename=save_path, overwrite=overwrite
        )

    @staticmethod
    def get_schema_generation_configuration(configs: Dict[str, Any]):
        schema_meta_data_file_dir = configs["dataset_folder"] + "/{}_tables.json".format(configs["evaluation_type"])
        dataset_schema_path = "./scalesql/dataset/bird_{}_light_schema.json".format(configs["evaluation_type"])
        column_meaning_path = configs["column_meaning_path"]
        database_execution_path = configs["dataset_folder"] + "/{}_databases/".format(
            configs["evaluation_type"]) + "{db}/{db}.sqlite"

        return dict(
            schema_meta_data_file_dir=schema_meta_data_file_dir,
            dataset_schema_path=dataset_schema_path,
            column_meaning_path=column_meaning_path,
            database_execution_path=database_execution_path
        )

    @staticmethod
    def light_schema_generation(schema_generation_configuration: Dict[str, Any]):
        def get_random_rows(db_path, table_name, column_name):
            query = f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL ORDER BY RANDOM() LIMIT 3;"

            cursor = get_cursor_from_path(db_path)
            cursor.execute(query)
            results = cursor.fetchall()
            if len(cursor.description) == 1:
                return [row[0] for row in results]
            else:
                return results

        def get_format_column_meaning(column_meaning_path):
            column_meaning_raw = read_json(column_meaning_path)
            column_meaning = {}
            for key, value in column_meaning_raw.items():
                db_id, table_name, column_name = key.split('|')
                if db_id not in column_meaning:
                    column_meaning[db_id] = {}
                if table_name not in column_meaning[db_id]:
                    column_meaning[db_id][table_name] = {}
                value = value.replace('#', '').replace('\n', ' ').strip()
                column_meaning[db_id][table_name][column_name] = value
            return column_meaning

        schema_meta_data_file_dir = schema_generation_configuration["schema_meta_data_file_dir"]
        db_path_template = schema_generation_configuration["database_execution_path"]
        column_meaning = get_format_column_meaning(schema_generation_configuration["column_meaning_path"])

        metadata = read_json(schema_meta_data_file_dir)
        logging.info(f"有 {len(metadata)} 个数据库的 schema需要产生")
        schema_dict = {}
        for data in metadata:
            table_list = {}
            db = data["db_id"]
            table_names = data["table_names_original"]
            for table in table_names:
                table_list[table] = dict(
                    table="",
                    columns=[],
                    primary_key=[],
                    foreign_key=[],
                )
            column_names = data["column_names_original"]
            column_types = data["column_types"]
            primary_keys = data["primary_keys"]
            primary_keys = [
                item
                for sublist in primary_keys
                for item in (sublist if isinstance(sublist, list) else [sublist])
            ]
            foreign_keys = data["foreign_keys"]

            for [table_id, column_name], type in zip(
                    column_names, column_types
            ):
                if table_id < 0:
                    continue
                table_name = table_names[table_id]
                samples = get_random_rows(
                    db_path=db_path_template.format(db=db),
                    table_name=table_name,
                    column_name=column_name,
                )
                processed_samples = []
                for sample in samples:
                    if isinstance(sample, str):
                        if len(str(sample)) <= 32:
                            processed_samples.append(sample)
                        else:
                            processed_samples.append(sample[:32] + "...")
                    else:
                        processed_samples.append(sample)
                samples = processed_samples

                column_desc = column_meaning[db][table_name].get(column_name, "")
                table_list[table_name]["columns"].append(
                    dict(name=column_name, type=type, description=column_desc, samples=samples)
                )

            for i in primary_keys:
                table_id, column_name = column_names[i]
                table_name = table_names[table_id]
                table_list[table_name]["primary_key"].append(column_name)

            for [column_id1, column_id2] in foreign_keys:
                table_id1, column_name1 = column_names[column_id1]
                table_id2, column_name2 = column_names[column_id2]
                table1, table2 = table_names[table_id1], table_names[table_id2]
                fk_desc = f"{table1}.{column_name1} = {table2}.{column_name2}"
                table_list[table1]["foreign_key"].append(fk_desc)
                table_list[table2]["foreign_key"].append(fk_desc)

            schema = ""
            for table, table_info in table_list.items():
                schema += (
                        LightSchema.create_schema(
                            database=db,
                            table=table,
                            columns=table_info["columns"],
                            primary_key=table_info["primary_key"],
                            foreign_key=table_info["foreign_key"],
                        ).replace("### Table description\n\n", "")
                        + "\n"
                )

            schema_dict[db] = schema
            logging.info(f"已经产生完成 {db} 的数据库的模式.")

        logging.info(f"已经产生完成 {len(schema_dict)} 个数据库的模式.")
        logging.info(f"打印第一个模式:\n{next(iter(schema_dict.values()))}")
        return schema_dict

    @staticmethod
    def main(configs: Dict[str, Any]) -> None:
        schema_generation_configuration = (
            SchemaGeneration.get_schema_generation_configuration(configs)
        )
        logging.info(f"schema_generation_configuration:\n {schema_generation_configuration}")

        light_schema = SchemaGeneration.light_schema_generation(
            schema_generation_configuration
        )

        save_or_append_json(
            data=light_schema, filename=schema_generation_configuration["dataset_schema_path"], overwrite=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation_type",
        required=True,
        type=str,
        choices=["dev", "test", "train"],
        default="test"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="scalesql/workflows/config/pipeline_config.yaml"
    )
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    for key, value in vars(args).items():
        if value is not None:
            configs[key] = value

    logging.info(f"configs:\n{configs}")

    llm_config = read_env()
    logging.info(llm_config)
    configs["llm_config"] = llm_config

    SchemaGeneration.main(configs)
