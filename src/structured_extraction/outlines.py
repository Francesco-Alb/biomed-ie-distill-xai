import os
from dotenv import load_dotenv
import time
from pathlib import Path

import pandas as pd
import datasets
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm

import instructor

# Ensure underlying dependencies are accessible
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import outlines
from outlines.inputs import Chat

_LOCAL_MODEL_CACHE = {}

load_dotenv()
hf_token = os.getenv("HF_TOKEN")


class OutlinesInstructorWrapper:
    """Wraps an Outlines Hugging Face model to mimic Instructor's functional signature."""
    def __init__(self, outlines_model):
        self.outlines_model = outlines_model

    def __call__(self, messages, response_model, temperature=0.0, **kwargs):

        chat_prompt = Chat(messages)
        
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        gen_kwargs = {"max_new_tokens": max_new_tokens}
        if temperature == 0.0:
            gen_kwargs["do_sample"] = False
            # DO NOT pass a temperature key here, or Transformers will throw a validation error.
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        
        json_str = self.outlines_model(
            chat_prompt,
            output_type=response_model,
            **gen_kwargs
        )
        
        # Parse the raw JSON string into a native Pydantic instance to support downstream .model_dump()
        return response_model.model_validate_json(json_str)


def _get_model(model_checkpoint: str, use_local_model: bool = False, verbose: bool = False):
    """Return (client, create) where one element is None depending on provider type.

    - client: an Instructor client with `.create()` when using provider APIs
    - create: a local OutlinesInstructorWrapper when using a native local Transformers model
    """
    global _LOCAL_MODEL_CACHE
    client = None
    create = None

    # Check if this exact model configuration is already active in VRAM
    cache_key = (model_checkpoint, use_local_model)
    if cache_key in _LOCAL_MODEL_CACHE:
        if verbose or use_local_model:
            print("ℹ️ Model already loaded in VRAM. Reusing active session...")
        return _LOCAL_MODEL_CACHE[cache_key]

    if not use_local_model:
        client = instructor.from_provider(
            model_checkpoint,
            mode=instructor.Mode.TOOLS,
        )
    else:
        print(f"ℹ️ Using local model via Outlines & Transformers: {model_checkpoint}")

        quant_config = None
        
        is_small_model = any(size_tag in model_checkpoint.lower() for size_tag in ["3b", "1.5b", "1b", "0.5b"])
        if not is_small_model:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            quantization_config = quant_config,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="sdpa",  # <-- Activates optimized hardware attention loops
            token=hf_token if hf_token else False,
        )
        hf_tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            token = hf_token if hf_token else False,
            )

        outlines_model = outlines.from_transformers(hf_model, hf_tokenizer)
        
        create = OutlinesInstructorWrapper(outlines_model)

    _LOCAL_MODEL_CACHE[cache_key] = (client, create)
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
            print("Starting full instructor/outlines pipeline...")

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