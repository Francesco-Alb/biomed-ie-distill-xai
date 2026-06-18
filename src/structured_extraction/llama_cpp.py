import os
import time
from pathlib import Path

import pandas as pd
import datasets
from datasets import Dataset, DatasetDict
from huggingface_hub import snapshot_download
from llama_cpp import Llama
from tqdm.auto import tqdm

import instructor


def _get_model(model_checkpoint: str, use_local_model: bool = False, verbose: bool = False):
    """Return (client, create) where one element is None depending on provider type.

    - client: an Instructor client with `.create()` when using provider APIs
    - create: a local callable (e.g. patched `llama.create_chat_completion_openai_v1`) when using a local model
    """
    client = None
    create = None

    if not use_local_model:
        client = instructor.from_provider(
            model_checkpoint,
            mode=instructor.Mode.TOOLS,
        )
    else:
        print(f"ℹ️ Using local model via llama-cpp: {model_checkpoint}")

        # Define pattern to ensures we only grab the 4-bit files, not the whole repo
        pattern = (
            model_checkpoint
            .lower()
            .split('/')[1]
            .replace('gguf', 'q4_k_m')
        )

        # Download all shards for the Q4_K_M model
        model_dir = snapshot_download(
            repo_id=model_checkpoint,
            allow_patterns=f"*{pattern}*",
        )

        # first_shard = sorted(os.listdir(model_dir))[0]
        # assert ("-00001-of-" in first_shard), "First shard must contain '-00001-of-'"

        # # Identify the FIRST shard: llama-cpp needs you to point to the -00001-of-00002 (first shard) file specifically
        # model_path = os.path.join(model_dir, first_shard)

        gguf_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.gguf')])
        if not gguf_files:
            raise FileNotFoundError(f"No GGUF files found matching pattern in {model_dir}")
        first_file = gguf_files[0]        
        model_path = os.path.join(model_dir, first_file)
        print(f"Model path: {model_path}")

        # Load the model normally
        llama = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_batch=512,
            n_ctx=1500, # 8192
            chat_format="qwen",
            verbose=verbose,
        )

        create = instructor.patch(
            create=llama.create_chat_completion_openai_v1,
            mode=instructor.Mode.JSON,
        )

    return client, create


def _get_create_callable(client, create):
    if create is not None:
        return create
    if client is not None:
        return lambda **kwargs: client.create(**kwargs)
    raise RuntimeError("No model callable available. Check your model_checkpoint and use_local_model settings.")


def get_response(
    df_to_structure: pd.DataFrame | datasets.dataset_dict.DatasetDict,
    id_col: str,
    text_col: str,
    model_checkpoint: str,
    structured_file: Path | None,
    structured_checkpoint_file: Path | None,
    response_model,
    instruction: str,
    temperature: float = 0.0,
    max_retries: int = 1,
    checkpoint_steps: int = 50,
    use_local_model: bool = False,
    n_rows_to_process: int|None = None,
    dataset_split: str = "train",
    verbose: bool = False
) -> pd.DataFrame:
    """
    Process a DataFrame or a `datasets.DatasetDict` and extract structured responses using an LLM.

    Args:
        df_to_structure: pandas DataFrame or `datasets.DatasetDict`
        id_col: str: Column name for the unique identifier of each row in the DataFrame
        text_col: column name of text content
        model_checkpoint: str: Path to the LLM model checkpoint
        structured_file: Path to save final structured data
        structured_checkpoint_file: Path to save checkpoint during processing
        response_model: Pydantic model for response validation
        instruction: System instruction for the model
        prefix: add a prefix to the text content
        temperature: Model temperature
        max_retries: Maximum retries for each request
        checkpoint_steps: Save checkpoint every N rows
        use_local_model: Whether using a local model
        n_rows_to_process: Maximum number of rows to process per run (if None all rows are processed)
        dataset_split: If `df_to_structure` is a DatasetDict, which split to use
        verbose: Whether to print (llama) progress messages

    Returns:
        pd.DataFrame: Processed DataFrame with structured responses
    """
    # Normalize input: allow DatasetDict / Dataset / pandas.DataFrame
    if isinstance(df_to_structure, DatasetDict):
        if dataset_split not in df_to_structure:
            raise ValueError(f"dataset_split '{dataset_split}' not found in DatasetDict")
        df = df_to_structure[dataset_split].to_pandas()
    elif isinstance(df_to_structure, Dataset):
        df = df_to_structure.to_pandas()
    else:
        df = df_to_structure.copy()

    if not id_col or id_col not in df.columns:
        raise ValueError("Make sure id_col is provided and exists in the DataFrame.")

    client, create = _get_model(
        model_checkpoint=model_checkpoint, 
        use_local_model=use_local_model,
        verbose=verbose
    )
    create_callable = _get_create_callable(client, create)

    if structured_file is not None and os.path.exists(structured_file):
        print("📂 Loading existing processed data from final file...")
        structure_only_df = pd.read_parquet(structured_file)
        print(f"✅ Loaded {len(structure_only_df)} rows.")
    else:
        print("ℹ️ Final processed/structured file not found. Checking for checkpoint...")

        processed_ids = set()
        current_processed_data = pd.DataFrame()

        if structured_checkpoint_file is not None and os.path.exists(structured_checkpoint_file):
            print(f"🔄 Resuming from checkpoint: {structured_checkpoint_file}")
            current_processed_data = pd.read_parquet(structured_checkpoint_file)
            processed_ids = set(current_processed_data[id_col].unique())
            print(f"✅ Loaded {len(processed_ids)} unique ids from checkpoint.")
        else:
            print("Starting full instructor pipeline...")

        remaining_df_to_structure = df[~df[id_col].isin(processed_ids)].copy()
        if n_rows_to_process is not None:
            remaining_df_to_structure = remaining_df_to_structure.head(n_rows_to_process)
        print(f"Total items to process: {len(remaining_df_to_structure)}")

        total_remaining = len(remaining_df_to_structure)
        structured_analysis_arguments = {
            "response_model": response_model,
            "temperature": temperature,
            "max_retries": max_retries,
            "stop": ["```"],
        }

        for i in tqdm(range(0, total_remaining, checkpoint_steps), desc="Generating Structured Data"):
            current_batch_df_to_process = remaining_df_to_structure.iloc[i: i + checkpoint_steps]
            current_batch_results_list = []

            for index, row in tqdm(current_batch_df_to_process.iterrows(), total=len(current_batch_df_to_process), leave=False):
                chemical = row['chemical']
                disease = row['disease']
                content = (
                    f"Analyze the relationship between the target chemical '{chemical}' and the target disease '{disease}' "
                    f"in the text below. Carefully evaluate the context for potential drug antagonism or linguistic polysemy "
                    f"as detailed in your system instructions.\n\n"
                    f"Text: {row[text_col]}"
                )

                item_id = row[id_col]

                if not use_local_model:
                    time.sleep(1)

                try:
                    analysis = create_callable(
                        **structured_analysis_arguments,
                        messages=[
                            {"role": "system", "content": instruction},
                            {"role": "user", "content": content},
                        ]
                    )

                    analysis_dict = analysis.model_dump()
                    analysis_dict[id_col] = item_id
                    analysis_dict["extraction_status"] = "SUCCESS"
                    current_batch_results_list.append(analysis_dict)

                except Exception as e:
                    print(f"❌ Error on row {index} ({id_col}: {item_id}): {e}")
                    current_batch_results_list.append({
                        id_col: item_id,
                        "extraction_status": f"FAILED: {str(e)[:100]}"
                    })

            if current_batch_results_list:
                current_batch_structured_df = pd.DataFrame(current_batch_results_list)
                if not current_processed_data.empty:
                    current_processed_data = pd.concat([current_processed_data, current_batch_structured_df])
                else:
                    current_processed_data = current_batch_structured_df

                if structured_checkpoint_file is not None:
                    current_processed_data.to_parquet(structured_checkpoint_file, index=False)
                    last_processed = current_batch_df_to_process.iloc[-1][id_col]
                    print(f"💾 Checkpoint saved at {structured_checkpoint_file} with {len(current_processed_data)} rows.")
                    print(f"Last item processed: {last_processed}")
            else:
                print(f"No results generated for batch starting at index {i}.")

        structure_only_df = current_processed_data
        if structured_file is not None:
            structure_only_df.to_parquet(structured_file, index=False)
            print(f"💾 Successfully saved {len(structure_only_df)} rows to {structured_file}")

    structured_df = pd.merge(df, structure_only_df, on=id_col, how='left')
    print(f"✅ Merged structured data with original DataFrame. Resulting DataFrame has {len(structured_df)} rows.")

    return structured_df