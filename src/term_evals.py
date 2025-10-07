import pandas as pd
import re
from collections import defaultdict

# --- Configuration ---
# Ensure your CSV file is named this and is in the same folder as the script.
FILE_PATH = '/data/test_dataset_result_v4r6.csv'
# This should be the name of the column containing the SQL queries.
QUERY_COLUMN = 'input_query'

def analyze_sql_dataset():
    """
    Loads a CSV, analyzes the SQL queries in a specified column,
    and prints a statistical breakdown of statements and clauses.
    """
    # --- Data Structures ---
    statement_counts = defaultdict(int)
    clause_counts = defaultdict(lambda: defaultdict(int))
    total_queries_analyzed = 0

    # --- Regex Patterns for Clauses ---
    # Using word boundaries (\b) to avoid matching substrings like 'SELECTion'
    CLAUSE_PATTERNS = {
        'WHERE': re.compile(r'\bWHERE\b', re.IGNORECASE),
        'JOIN': re.compile(r'\bJOIN\b', re.IGNORECASE),
        'GROUP BY': re.compile(r'\bGROUP BY\b', re.IGNORECASE),
        'ORDER BY': re.compile(r'\bORDER BY\b', re.IGNORECASE),
        'HAVING': re.compile(r'\bHAVING\b', re.IGNORECASE),
        'LIMIT': re.compile(r'\bLIMIT\b', re.IGNORECASE),
        'UNION': re.compile(r'\bUNION\b', re.IGNORECASE),
        'INTERSECT': re.compile(r'\bINTERSECT\b', re.IGNORECASE),
        'EXCEPT': re.compile(r'\bEXCEPT\b', re.IGNORECASE),
    }

    try:
        # 1. Load the dataset
        df = pd.read_csv(FILE_PATH)
        print(f"Successfully loaded {len(df)} rows from '{FILE_PATH}'. Analyzing queries...")

        # 2. Analyze each query
        for query in df[QUERY_COLUMN]:
            if not isinstance(query, str):
                continue
            
            total_queries_analyzed += 1
            
            # Determine statement type
            stmt_type = None
            normalized_query = query.strip().upper()
            if normalized_query.startswith('SELECT'):
                stmt_type = 'SELECT'
            elif normalized_query.startswith('INSERT'):
                stmt_type = 'INSERT'
            elif normalized_query.startswith('UPDATE'):
                stmt_type = 'UPDATE'
            elif normalized_query.startswith('DELETE'):
                stmt_type = 'DELETE'
            
            if not stmt_type:
                continue
            
            statement_counts[stmt_type] += 1
            
            # 3. Identify and count clauses using regex
            for clause_name, pattern in CLAUSE_PATTERNS.items():
                matches = pattern.findall(query)
                if matches:
                    clause_counts[stmt_type][clause_name] += len(matches)

            # Count subqueries as a special case
            subquery_count = len(re.findall(r'\(SELECT', query, re.IGNORECASE))
            if subquery_count > 0:
                clause_counts[stmt_type]['Subquery'] += subquery_count

        # --- 4. Print Results ---
        print("\n" + "="*40)
        print("   SQL Query Dataset Analysis Results")
        print("="*40 + "\n")
        print(f"Total Queries Analyzed: {total_queries_analyzed}\n")

        for stmt, count in sorted(statement_counts.items()):
            print(f"Statement Type: {stmt} (Total Found: {count})")
            print("-" * 35)
            if clause_counts[stmt]:
                for clause, clause_count in sorted(clause_counts[stmt].items(), key=lambda item: item[1], reverse=True):
                    percentage = (clause_count / count) * 100
                    print(f"  - {clause:<12}: {clause_count:<5} ({percentage:.1f}%)")
            else:
                print("  - No complex clauses found.")
            print("\n")

    except FileNotFoundError:
        print(f"Error: The file '{FILE_PATH}' was not found.")
        print("Please make sure the CSV file is in the same directory as this script.")
    except KeyError:
        print(f"Error: The column '{QUERY_COLUMN}' was not found in the CSV file.")
        print(f"Please check your CSV and update the QUERY_COLUMN variable if needed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    analyze_sql_dataset()