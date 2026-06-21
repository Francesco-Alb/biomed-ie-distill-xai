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

from src.structured_extraction.pydantic_schema import PairwiseExtraction, GlobalExtraction

_LOCAL_MODEL_CACHE = {}

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

 
class OutlinesInstructorWrapper:
    """Wraps an Outlines Hugging Face model to mimic Instructor's functional signature."""
    def __init__(self, outlines_model):
        self.outlines_model = outlines_model

    def __call__(self, messages, response_model, temperature=0.0, **kwargs):

        chat_prompt = Chat(messages)
        
        max_new_tokens = kwargs.get("max_new_tokens", 256)
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


def _get_model(
        model_checkpoint: str,
        model_source: str = "api",
        quantize_model: bool = False,
        tensor_parallel_size: int = 2,
        ):
    """Return (client, create) where one element is None depending on provider type.

    - client: an Instructor client with `.create()` when using provider APIs
    - create: a local OutlinesInstructorWrapper when using Outlines local models
    """
    global _LOCAL_MODEL_CACHE
    client = None
    create = None

    valid_sources = {"api", "transformers", "vllm_offline"}
    if model_source not in valid_sources:
        raise ValueError(
            f"model_source must be one of {sorted(valid_sources)}, got '{model_source}'"
        )
    
    if quantize_model and model_source != "api":
        pre_quantized_keywords = ['awq', 'gptq', 'gguf', 'int4', 'int8', '4bit', '8bit', 'quantized']
        is_pre_quantized_model = any(keyword in model_checkpoint.lower() for keyword in pre_quantized_keywords)
        assert not (quantize_model and is_pre_quantized_model), (
            f"Conflict detected! You set quantized_model=True, but the model checkpoint "
            f"'{model_checkpoint}' appears to already be pre-quantized. "
            f"Set quantized_model to False to use this pre-quantized model safely."
        )
        quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

    # Check if this exact model configuration is already active in VRAM
    cache_key = (model_checkpoint, model_source, quantize_model, tensor_parallel_size)
    if cache_key in _LOCAL_MODEL_CACHE:
        if model_source != "api":
            print("ℹ️ Model already loaded in VRAM. Reusing active session...")
        return _LOCAL_MODEL_CACHE[cache_key]

    if model_source == "api":
        client = instructor.from_provider(
            model_checkpoint,
            mode=instructor.Mode.TOOLS,
        )

    elif model_source == "transformers":
        print(f"ℹ️ Using local model via Outlines & Transformers: {model_checkpoint}")

        model_kwargs = {
            "token": hf_token if hf_token else None,
            "torch_dtype": "auto",
            "device_map": "auto",
            "attn_implementation": "sdpa",  # Activates optimized hardware attention loops
            "low_cpu_mem_usage": True,       # Prevents system RAM spikes
        }

        if quantize_model:
            model_kwargs["quantization_config"] = quantization_config

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            **model_kwargs,
        )
        hf_tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            token=hf_token if hf_token else None,
        )

        outlines_model = outlines.from_transformers(hf_model, hf_tokenizer)
        create = OutlinesInstructorWrapper(outlines_model)
    else:
        print(f"ℹ️ Using local model via Outlines vLLM Offline: {model_checkpoint}")

        model_kwargs = {
            "token": hf_token if hf_token else None,
            "tensor_parallel_size": tensor_parallel_size,
        }

        if quantize_model:
            model_kwargs["quantization_config"] = quantization_config

        outlines_model = outlines.from_vllm_offline(
            model_checkpoint,
            **model_kwargs
        )
        create = OutlinesInstructorWrapper(outlines_model)

    _LOCAL_MODEL_CACHE[cache_key] = (client, create)
    return client, create


def _get_create_callable(client, create):
    if create is not None:
        return create
    if client is not None:
        return lambda **kwargs: client.create(**kwargs)
    raise RuntimeError("No model callable available. Check your model_checkpoint and model_source settings.")


def get_response(
    id_col: str,
    text_col: str,
    model_checkpoint: str,
    instruction: str,
    df_to_structure: pd.DataFrame | datasets.dataset_dict.DatasetDict,
    df_exploded_reference: pd.DataFrame | None = None, # Required ONLY if is_macro=True
    structured_file: Path | None = None,
    structured_checkpoint_file: Path | None = None,
    is_macro: bool = False,
    quantize_model: bool = False,
    model_source: str = "api",
    tensor_parallel_size: int = 2,
    temperature: float = 0.0,
    max_retries: int = 1,
    max_tokens: int = 0,
    checkpoint_steps: int = 50,
    sleep_time: int = 5,
    n_rows_to_process: int | None = None,
    dataset_split: str = "train",
    drop_nans_from_checkpoint: bool = False
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

    response_model = GlobalExtraction if is_macro else PairwiseExtraction
    if max_tokens == 0:
        max_tokens = 4096 if is_macro else 1024

    if not id_col or id_col not in df.columns:
        raise ValueError("Make sure id_col is provided and exists in the DataFrame.")

    client, create = _get_model(
        model_checkpoint=model_checkpoint,
        model_source=model_source,
        quantize_model=quantize_model,
        tensor_parallel_size=tensor_parallel_size,
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
            if drop_nans_from_checkpoint:
                print("🗑️ Dropping rows with NaN values from checkpoint before resuming.")
                current_processed_data = current_processed_data.dropna()
            processed_ids = set(current_processed_data[id_col].unique())
            print(f"✅ Loaded {len(processed_ids)} unique ids from checkpoint.")
        else:
            print("Starting full instructor/outlines pipeline...")

        remaining_df = df[~df[id_col].isin(processed_ids)].copy()
        if n_rows_to_process is not None:
            remaining_df = remaining_df.head(n_rows_to_process)
        print(f"Total items to process: {len(remaining_df)}")

        structured_analysis_arguments = {
            "response_model": response_model,
            "temperature": temperature,
            "max_retries": max_retries,
            "stop": ["```"],
            "max_tokens": max_tokens,
        }

        # 🔄 Automatically adjust token key when switching to local HuggingFace transformers path
        if model_source == "transformers":
            structured_analysis_arguments["max_new_tokens"] = structured_analysis_arguments.pop("max_tokens")

        for i in tqdm(range(0, len(remaining_df), checkpoint_steps), desc="Generating Structured Data"):
            batch_df = remaining_df.iloc[i: i + checkpoint_steps]
            batch_results = []

            for index, row in tqdm(batch_df.iterrows(), total=len(batch_df), leave=False):
                item_id = row[id_col]
                
                # --- BRANCH A: MACRO LOGIC (Abstract-Level) ---
                if is_macro:
                    if df_exploded_reference is None:
                        raise ValueError("df_exploded_reference must be provided when is_macro=True")
                
                    # Fetch the exact candidate pairs assigned to this abstract
                    doc_candidates = df_exploded_reference[df_exploded_reference[id_col] == item_id]
                    candidate_list_str = "\n".join([
                        f"- Chemical: {r['chemical']} | Disease: {r['disease']}" 
                        for _, r in doc_candidates.iterrows()
                    ])

                    content = (
                        f"Abstract Text:\n{row[text_col]}\n\n"
                        f"CRITICAL ASSIGNMENT: Evaluate the relationship status for ONLY these explicit candidate pairs:\n"
                        f"{candidate_list_str}\n\n"
                        f"Do not evaluate pairs outside of this list."
                    )
                
                # --- BRANCH B: MICRO LOGIC (Pair-by-Pair) ---
                else:
                    content = (
                        f"TARGET CHEMICAL TO EVALUATE: {row['chemical']}\n"
                        f"TARGET DISEASE TO EVALUATE: {row['disease']}\n\n"
                        f"Text:\n{row['masked_text']}\n\n"
                        f"Question: What is the direct relationship of the TAGGED <chemical>{row['chemical']}</chemical> "
                        f"towards the TAGGED <disease>{row['disease']}</disease>?"
                    )


                if model_source == "api":
                    time.sleep(sleep_time)

                try:
                    analysis = create_callable(
                        **structured_analysis_arguments,
                        messages=[
                            {"role": "system", "content": instruction},
                            {"role": "user", "content": content},
                        ]
                    )

                    if is_macro:
                        # Flatten out the array of items generated for this single abstract
                        for rel in analysis.relationships:
                            rel_dict = rel.model_dump()
                            rel_dict[id_col] = item_id
                            rel_dict["extraction_status"] = "SUCCESS"
                            batch_results.append(rel_dict)
                    else:
                        # Single row processing
                        rel_dict = analysis.model_dump()
                        rel_dict[id_col] = item_id
                        rel_dict["extraction_status"] = "SUCCESS"
                        batch_results.append(rel_dict)

                except Exception as e:
                    print(f"❌ Error on row {index} ({id_col}: {item_id}): {e}")
                    batch_results.append({
                        id_col: item_id,
                        "extraction_status": f"FAILED: {str(e)[:100]}"
                    })

            if batch_results:
                batch_res_df = pd.DataFrame(batch_results)
                if not current_processed_data.empty:
                    current_processed_data = pd.concat([current_processed_data, batch_res_df])
                else:
                    current_processed_data = batch_res_df

                if structured_checkpoint_file is not None:
                    current_processed_data.to_parquet(structured_checkpoint_file, index=False)
                    last_processed = batch_df.iloc[-1][id_col]
                    print(f"💾 Checkpoint saved at {structured_checkpoint_file} with {len(current_processed_data)} rows.")
                    print(f"Last item processed: {last_processed}")
            else:
                print(f"No results generated for batch starting at index {i}.")

        structure_only_df = current_processed_data
        if structured_file is not None:
            structure_only_df.to_parquet(structured_file, index=False)
            print(f"💾 Successfully saved {len(structure_only_df)} rows to {structured_file}")

    if is_macro:
        structured_df = pd.merge(
            df_exploded_reference, 
            structure_only_df, 
            on=[id_col, 'chemical', 'disease'], 
            how='left'
        )
    else:
        structured_df = pd.merge(
            df, 
            structure_only_df, 
            on=[id_col, 'chemical', 'disease'], 
            how='left'
        )

    
    print(f"✅ Merged structured data with original DataFrame. Resulting DataFrame has {len(structured_df)} rows.")
    return structured_df