#!/usr/bin/env python3
"""
Extracted from experiments/P3T2Q_benchmark_builder.ipynb
section: ## Modular implementation

Generates access_control.json files for benchmark DB folders.
"""

import os
import json
import re 
import unicodedata 
import inflect 
import spacy
import random


class AccessControlRuleGenerator:
    # Keep important short tokens stable
    KEEP_COL = {
        "id", "ids", "dob", "uuid", "uid", "ssn", "ip",
        "frpm", "nslp", "nces", "calpads",
        "iga", "igg", "igm", "ana", "lac", "crp",
        "ssa", "ssb", "soc", "rf", "ua", "tp",
        "glu", "hgb", "ldh", "gpt", "alp",
        "cds", "rbc", "wbc", "rvvt", "kct",
    }

    DROP_COL = {"the", "a", "an", "of", "and", "or", "to", "for"}

    _CAMEL_COL = re.compile(
        r"(?<=[a-z0-9])(?=[A-Z][a-z])"
        r"|(?<=[A-Z])(?=[A-Z][a-z])"
        r"|(?<=[a-z0-9])(?=[A-Z]{2,}\b)"
    )

    _TOKEN_COL = re.compile(r"[A-Za-z]+[0-9]+|[0-9]+|[A-Za-z]+")
    _ROMAN_COL = re.compile(r"^(?=[ivxlcdm]+$)[ivxlcdm]+$", re.IGNORECASE)

    _PROTECT_SINGULAR_SUFFIXES = (
        "osis", "itis", "sis",   # diagnosis, thrombosis, analysis, etc.
        "us", "as", "is"         # many singulars end with these too
    )

    KEEP_TAB = {"frpm", "zip", "zipcode", "zip_code"}

    _CAMEL_TAB = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
    _TOKEN_TAB = re.compile(r"[A-Za-z]+[0-9]+|[0-9]+[A-Za-z]+|[0-9]+|[A-Za-z]+")

    # define access control informations 
    ROLES = ["admin", "analyst", "staff", "public"]

    BASE_POLICIES = [
        {
        "id": "col_view_S1",
        "effect": "deny",
        "action": "read",
        "scope": "view",
        "level": "column",
        "categories": ["S1"],
        "roles": ["public"],
        },
        {
        "id": "col_view_S2",
        "effect": "deny",
        "action": "read",
        "scope": "view",
        "level": "column",
        "categories": ["S2"],
        "roles": ["public","staff"]
        },
        {
        "id": "col_view_S3",
        "effect": "deny",
        "action": "read",
        "scope": "view",
        "level": "column",
        "categories": ["S3"],
        "roles": ["public","staff","analyst"]
        },
        {
        "id": "col_proc_S2",
        "effect": "deny",
        "action": "read",
        "scope": "process",
        "level": "column",
        "categories": ["S2"],
        "roles": ["public"]
        },
        {
        "id": "col_proc_S3",
        "effect": "deny",
        "action": "read",
        "scope": "process",
        "level": "column",
        "categories": ["S3"],
        "roles": ["public","staff"]
        },
        {
        "id": "table_view_S2",
        "effect": "deny",
        "action": "read",
        "scope": "view",
        "level": "table",
        "categories": ["S2"],
        "roles": ["public","staff"]
        },
        {
        "id": "table_view_S1",
        "effect": "deny",
        "action": "read",
        "scope": "view",
        "level": "table",
        "categories": ["S1"],
        "roles": ["public"]
        },
        {
        "id": "table_process_S2",
        "effect": "deny",
        "action": "read",
        "scope": "process",
        "level": "table",
        "categories": ["S2"],
        "roles": ["public"]
        }
    ]
    ROW_POLICY = {
      "id": "row_view_public",
      "effect": "conditional",
      "action": "read",
      "scope": "view",
      "level": "row",
      "roles": ["public","staff","analyst"],
      "table": "Patient",
    #   "predicate": { "column": "SEX", "op": "=", "value_type": "category", "value": "S1" }
    }

    def __init__(self):
        # keyword dictionary and sensitive table list can be further expanded and refined based on domain knowledge and data profiling results
        self.highly_sensitive_tables = ["client", "user", "account", "transaction", "payment", "order", "customer", "employee", "patient", "medical_record", "diagnosis", "symptom", "examination"]
        self.fairly_sensitive_tables = ["card", "loan", "bank", "bond", "income", "laboratory", "transaction", "payment"]

        self.column_keyword_dict ={
            "S3": ["password", "passwd", "pwd", "secret", "token", "apikey", "auth", "oauth", "session", "cookie", "hash", "salt", "pin", "ssn", "social", "sin", "nin", "passport", "license", "tax", "tin", "credit", "cc", "cvv", "cvc", "card", "iban", "swift", "routing", "account", "diagnosis"],
            "S2": ["email", "mail","zip+code", "postal", "address", "phone", "mobile", "cell", "tel", "fax", "first+name", "last+name", "full+name", "surname", "diagnosis", "symptom", "examintation", "dna", "hgb", "wbc", "rbc", "glu", "bank", "dob", "birth", "birthday"], 
            "S1": ["gender", "sex", "race", "nationality", "height", "weight", "city", "state", "country", "district"]
        }

        self.sensitive_table_names = ["client", "user", "account", "transaction", "payment", "order", "customer", "employee", "patient", "medical_record", "diagnosis", "symptom", "examination"]

        self._inflect = inflect.engine()

    def _ascii_fold(self, s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        return "".join(ch for ch in s if not unicodedata.combining(ch))

    def _preprocess(self, text: str) -> str:
        s = self._ascii_fold(text).strip()
        s = s.replace("*", " ")
        s = re.sub(r"[_/.,:;()\[\]{}%]+", " ", s)
        s = re.sub(r"-+", " ", s)
        s = self._CAMEL_COL.sub(" ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _is_code_like(self, original_token: str) -> bool:
        # Keep mixed alphanumerics intact (A4, q2, enroll12, SC170, NumGE1500)
        if any(c.isalpha() for c in original_token) and any(c.isdigit() for c in original_token):
            return True
        # Short all-caps codes (IGA, CRP, RBC, RVVT, etc.)
        if original_token.isupper() and 2 <= len(original_token) <= 10:
            return True
        return False

    def _safe_singularize(self, t: str) -> str:
        """
        Very conservative singularization:
        - only apply to pure alphabetic tokens
        - only apply when token ends with 's'
        - block known singular suffix families like -osis/-sis/-itis
        - accept only if inflect is confident it's plural
        """
        if not (t.isalpha() and len(t) >= 5):
            return t
        if not t.endswith("s"):
            return t
        if t.endswith(self._PROTECT_SINGULAR_SUFFIXES):
            return t

        sing = self._inflect.singular_noun(t)
        # inflect returns False if it's already singular; otherwise returns the singular form
        if sing:
            return sing
        return t

    def _normalize_schema_noun_phrase(self, text: str) -> str:
        if not text:
            return ""

        s = self._preprocess(text)
        raw_tokens = self._TOKEN_COL.findall(s)
        if not raw_tokens:
            return ""

        out = []
        for raw in raw_tokens:
            t = raw.lower()

            if t in self.DROP_COL:
                continue

            # Preserve identifiers/codes and roman numerals
            if t in self.KEEP_COL or self._is_code_like(raw) or self._ROMAN_COL.match(t):
                out.append(t)
                continue

            t = self._safe_singularize(t)
            out.append(t)

        result = " ".join(out)

        # Stitch accidental splits if any upstream created them
        result = re.sub(r"\b(i)\s+(d)\b", r"\1\2", result, flags=re.IGNORECASE)
        result = re.sub(r"\b([a-z])\s+(\d+)\b", r"\1\2", result)

        return re.sub(r"\s+", " ", result).strip()  

    def _normalize_table_name(self, name: str) -> str:
        """
        Normalize a table name into a canonical noun phrase.
        Examples:
        lapTimes -> lap time
        transactions_1k -> transaction 1k
        zip_code -> zip code
        constructors -> constructor
        """
        if not name:
            return ""

        s = self._ascii_fold(name).strip()
        s = re.sub(r"[_\-]+", " ", s)
        s = self._CAMEL_TAB.sub(" ", s)
        s = re.sub(r"\s+", " ", s).strip()
        raw_tokens = self._TOKEN_TAB.findall(s)
        if not raw_tokens:
            return ""

        out = []
        for raw in raw_tokens:
            t = raw.lower()

            # keep important abbreviations
            if t in self.KEEP_TAB:
                out.append(t.replace("_", " "))
                continue

            # singularize alphabetic tokens (table names are almost always nouns)
            if t.isalpha() and len(t) >= 3:
                sing = self._inflect.singular_noun(t)
                if sing:
                    t = sing

            out.append(t)

        # normalize "zip_code" variants
        phrase = " ".join(out)
        phrase = phrase.replace("zip_code", "zip code").replace("zipcode", "zip code")
        phrase = re.sub(r"\s+", " ", phrase).strip()
        return phrase

    def _update_dedup(self, dst: dict, src: dict):
        for k, v in src.items():
            if k in dst: 
                v2 = dst[k]
                dedup_v_arg = int(v[1]) > int(v2[1])
                if dedup_v_arg:
                    dst[k] = v
                else:
                    dst[k] = v2
            else:
                dst[k] = v
    
    def _categorize_column(self, column_name, column_keyword_dict, sensitive_table=False):
        column_keys = column_name.split()
        for category, keywords in column_keyword_dict.items():
            for keyword in keywords:
                if sensitive_table: 
                    if "name" in column_keys:
                        return "S2"
                    elif "id" in column_keys:
                        return "S2"
                
                if "id" in column_keys:
                        return "S1"
                    
                if "+" in keyword:
                    keyword_parts = keyword.split("+")
                    if all(part in column_keys for part in keyword_parts):
                        return category
                elif keyword in column_keys:
                        return category
        return "S0" 
    
    def column_sensitivity_by_table(self, column_names, table_name, column_keyword_dict, sensitive_table_names):
        sensitive_table = self._normalize_table_name(table_name) in sensitive_table_names
        column_category_dict = {}
        for col in column_names:
            if col == "*":
                continue
            
            normalized_col = self._normalize_schema_noun_phrase(col)
            category = self._categorize_column(normalized_col, column_keyword_dict, sensitive_table)
            column_category_dict[col] = category

        return column_category_dict
    

    def database_access_control_rule_generation(self, db):
        # print(f"Processing database: {db['db_id']}")
        # print("Original column names:", [col[1] for col in db['column_names_original']])
        # print("Normalized column names:", [normalize_schema_noun_phrase(col[1]) for col in db['column_names_original']])
        # table level access control rule generation     
        table_category_dict = {"S2": [], "S1": [], "S0": []}
        for table in db['table_names']:
            normalized_table = self._normalize_table_name(table)
            if normalized_table in self.highly_sensitive_tables:
                table_category_dict["S2"].append(table)
            elif normalized_table in self.fairly_sensitive_tables:
                table_category_dict["S1"].append(table)
            else:
                table_category_dict["S0"].append(table)

        # column level access control rule generation 
        row_level_access_control_rules = []
        table_column_category_dict = {}
        for table in db['table_names']:
            column_names = [col[1] for col in db['column_names'] if col[0] == db['table_names'].index(table)]
            category_column_dict = self.column_sensitivity_by_table(column_names, table, self.column_keyword_dict, self.sensitive_table_names)
            self._update_dedup(table_column_category_dict, category_column_dict)

            # row level acces control rule generation for "S2" column at "highly sensitive" tables 
            if table in table_category_dict["S2"]:
                for col, category in category_column_dict.items():
                    if category == "S1":
                        # add column by random change 
                        # if random.random() < 0.5:
                            row_level_access_control_rules.append({
                                "table": table,
                                "column": col,
                                "operation": "= type",
                                "category": category
                            })

        # merge all column by category 
        merged_category_column_dict = {}
        for col, category in table_column_category_dict.items():
            if category not in merged_category_column_dict:
                merged_category_column_dict[category] = []
            merged_category_column_dict[category].append(col)


        return table_category_dict, merged_category_column_dict, row_level_access_control_rules

    def run(self, base_dir):
        for db in os.listdir(base_dir):
            if db.startswith(".") or db.endswith(".json") or db.endswith(".md"):
                continue

            with open(os.path.join(base_dir, db, "schema.json"), "r") as f:
                db_schema = json.load(f)
            
            table_category_dict, merged_category_column_dict, row_level_access_control_rules = self.database_access_control_rule_generation(db_schema)

            # policy set creation
            policy_set = self.BASE_POLICIES.copy()
            if len(row_level_access_control_rules) > 0:
                for row_acc in row_level_access_control_rules:
                    row_policy = self.ROW_POLICY.copy()
                    row_policy['predicate'] = row_acc 
                    policy_set.append(row_policy)
                
            # building access control file 
            access_control_file = {
                "db_id": db_schema["db_id"],
                "classification": {
                    "table": table_category_dict,
                    "column": merged_category_column_dict
                },
                "policies": policy_set
            }

            with open(os.path.join(base_dir, db, "access_control.json"), "w") as f:
                json.dump(access_control_file, f, indent=4)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate access_control.json files from schema.json in benchmark DB directories."
    )
    parser.add_argument(
        "--base-dir",
        default="data/P3T2Q_benchmark/v0",
        help="Base directory containing one folder per DB (default: data/P3T2Q_benchmark/v0).",
    )
    args = parser.parse_args()

    acl_generator = AccessControlRuleGenerator()
    acl_generator.run(args.base_dir)


if __name__ == "__main__":
    main()
