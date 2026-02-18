CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate ScaleSQL

unzip ./nltk_data.zip -d /root/
my_config_path=scalesql/workflows/config/pipeline_config.yaml
evaluation_type=test
python -m scalesql.workflows.build_contents_bm25_index --config_path ${my_config_path} --evaluation_type ${evaluation_type}
python -m scalesql.workflows.ddl_schema_generation \
  --config_path ${my_config_path} \
  --evaluation_type ${evaluation_type}