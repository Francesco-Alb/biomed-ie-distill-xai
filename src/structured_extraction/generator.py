import os
from dotenv import load_dotenv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import datasets
from datasets import Dataset, DatasetDict
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

import instructor

# Ensure underlying dependencies are accessible
import torch
import outlines
from outlines.inputs import Chat

from src.structured_extraction.pydantic_schema import PairwiseExtraction, GlobalExtraction
from src.utils import verbose_print

_LOCAL_MODEL_CACHE = {}

load_dotenv()
hf_token = os.getenv("HF_TOKEN")


class OutlinesInstructorWrapper:
    """Wraps an Outlines Hugging Face model to mimic Instructor's functional signature."""
    def __init__(self, outlines_model, verbose: bool = False):
        self.outlines_model = outlines_model
        self.verbose = verbose

    def __call__(self, messages, output_type, **kwargs):

        chat_prompt = Chat(messages)
        
        json_str = self.outlines_model(
            chat_prompt,
            output_type=output_type,
            **kwargs
        )

        if self.verbose:
            verbose_print(
                self.verbose,
                "[DEBUG] Outlines kwargs:",
                kwargs,
                "[DEBUG] Response type:",
                type(json_str),
                "[DEBUG] Raw output:",
                json_str,
            )
        
        # Parse the raw JSON string into a native Pydantic instance to support downstream .model_dump()
        return output_type.model_validate_json(json_str)


def _get_model(
        model_checkpoint: str,
        model_source: str = "api",
        quantize_model: bool = False,
        tensor_parallel_size: int = 2,
        verbose: bool = False,
        ):
    """Return (client, create) where one element is None depending on provider type.

    - client: an Instructor client with `.create()` when using provider APIs
    - create: a local OutlinesInstructorWrapper when using Outlines local models
    """
    global _LOCAL_MODEL_CACHE
    client = None
    create = None

    valid_sources = {"api", "llama_cpp", "transformers", "vllm_offline"}
    if model_source not in valid_sources:
        raise ValueError(
            f"model_source must be one of {sorted(valid_sources)}, got '{model_source}'"
        )

    quantization_config = None
    if quantize_model and model_source != "api":
        from transformers import BitsAndBytesConfig

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

    elif model_source == "llama_cpp":
        from llama_cpp import Llama

        print(f"ℹ️ Using local model via Outlines & llama-cpp: {model_checkpoint}")

        pattern = (
            model_checkpoint
            .lower()
            .split('/')[1]
            .replace('gguf', 'q4_k_m')
        )

        model_dir = snapshot_download(
            repo_id=model_checkpoint,
            allow_patterns=f"*{pattern}*",
        )

        gguf_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.gguf')])
        if not gguf_files:
            raise FileNotFoundError(f"No GGUF files found matching pattern in {model_dir}")
        first_file = gguf_files[0]
        model_path = os.path.join(model_dir, first_file)
        print(f"Model path: {model_path}")

        outlines_model = outlines.from_llamacpp(
            Llama(
                model_path=model_path,
                n_gpu_layers=-1,
                n_batch=512,
                n_ctx=4096,
                # chat_format="qwen",
                verbose=False,
            )
        )

        create = OutlinesInstructorWrapper(outlines_model, verbose=verbose)

    elif model_source == "transformers":
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
        create = OutlinesInstructorWrapper(outlines_model, verbose=verbose)

    elif model_source == "vllm_offline":
        from vllm import LLM

        print(f"ℹ️ Using local model via Outlines vLLM Offline: {model_checkpoint}")

        llm = LLM(
            model=model_checkpoint,
            tensor_parallel_size=tensor_parallel_size,
        )

        model_kwargs = {
            "vllm_model": llm,
        }

        if quantize_model:
            model_kwargs["quantization_config"] = quantization_config

        outlines_model = outlines.from_vllm_offline(
            model_checkpoint,
            **model_kwargs,
        )
        create = OutlinesInstructorWrapper(outlines_model, verbose=verbose)

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
    instruction: str,
    df_to_structure: pd.DataFrame | datasets.dataset_dict.DatasetDict,
    model_checkpoint: str,
    df_exploded_reference: pd.DataFrame | None = None, # Required ONLY if is_macro=True
    structured_file: Path | None = None,
    structured_checkpoint_file: Path | None = None,
    is_macro: bool = False,
    quantize_model: bool = False,
    model_source: str = "api",
    tensor_parallel_size: int = 2,
    temperature: float = 0.0,
    max_retries: int = 1,
    max_tokens: int|str = "auto",
    checkpoint_steps: int = 25,
    sleep_time: int = 5,
    n_rows_to_process: int = 50,
    dataset_split: str = "train",
    drop_invalid_from_checkpoint: bool = False,
    verbose: bool = False,
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

    output_type = GlobalExtraction if is_macro else PairwiseExtraction

    if not id_col or id_col not in df.columns:
        raise ValueError("Make sure id_col is provided and exists in the DataFrame.")

    if is_macro and df_exploded_reference is None:
            raise ValueError("df_exploded_reference must be provided when is_macro=True")

    client, create = _get_model(
        model_checkpoint=model_checkpoint,
        model_source=model_source,
        quantize_model=quantize_model,
        tensor_parallel_size=tensor_parallel_size,
        verbose=verbose,
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
            if drop_invalid_from_checkpoint:
                print("🗑️ Dropping rows with NaN values from checkpoint before resuming.")
                current_processed_data = current_processed_data.dropna(
                    subset=[id_col, "weak_label", "extraction_status"]
                )
                if "extraction_status" in current_processed_data.columns:
                    status_ok = (
                        current_processed_data
                        .groupby(id_col)["extraction_status"]
                        .apply(lambda statuses: all(status == "SUCCESS" for status in statuses))
                    )
                    processed_ids = set(status_ok[status_ok].index)
                else:
                    processed_ids = set(current_processed_data[id_col].unique())
            else:
                processed_ids = set(current_processed_data[id_col].unique())
            print(f"✅ Loaded {len(processed_ids)} unique ids from checkpoint.")
        else:
            print("Starting full instructor/outlines pipeline...")

        remaining_df = df[~df[id_col].isin(processed_ids)].copy()

        if n_rows_to_process <= 0:
            raise ValueError("n_rows_to_process must be a positive integer.")
        
        remaining_df = remaining_df.head(n_rows_to_process)

        print(f"Total items to process: {len(remaining_df)}")

        max_tokens_arg = "max_tokens"
        gen_kwargs = {
            # "max_retries": max_retries,
            # "stop": ["```"],
            "output_type": output_type,
            "temperature": temperature,
        }

        if model_source == "transformers":
            max_tokens_arg = "max_new_tokens"

            if temperature == 0.0:
                gen_kwargs["do_sample"] = False
                gen_kwargs.pop("temperature") # Can't use both temperature = 0.0 and  "do_sample" = False; Transformers will throw a validation error.
            else:
                gen_kwargs["do_sample"] = True

        for i in tqdm(range(0, len(remaining_df), checkpoint_steps), desc="Generating Structured Data"):
            batch_df = remaining_df.iloc[i: i + checkpoint_steps]
            batch_results = []

            if verbose:
                verbose_print(
                    verbose,
                    "\n" + "=" * 60,
                    f"Batch {i // checkpoint_steps + 1} | rows {i} to {i + len(batch_df) - 1}",
                    "=" * 60,
                )

            for index, row in tqdm(batch_df.iterrows(), total=len(batch_df), leave=False):
                item_id = row[id_col]

                if verbose:
                    verbose_print(
                        verbose,
                        f"⏳ Processing row {index} | {id_col}={item_id}",
                        "-" * 60,
                    )
                
                # --- BRANCH A: MACRO LOGIC (Abstract-Level) ---
                if is_macro:
                
                    # Fetch the exact candidate pairs assigned to this abstract
                    doc_candidates = df_exploded_reference[df_exploded_reference[id_col] == item_id]
                    candidate_list = [
                        f"- Chemical: {r['chemical']} | Disease: {r['disease']}" 
                        for _, r in doc_candidates.iterrows()
                    ]
                    candidate_list_str = "\n".join(candidate_list)
                    n_candidates = len(candidate_list)

                    current_max_tokens = min(4096, 512 + (n_candidates * 300)) if max_tokens == "auto" else max_tokens
                    gen_kwargs[max_tokens_arg] = current_max_tokens

                    if verbose:
                        verbose_print(
                            verbose,
                            f"Macro mode detected. Candidate count: {n_candidates}",
                            f"Max tokens set to: {current_max_tokens}",
                            "Candidate list:\n" + candidate_list_str,
                        )

                    content = (
                        f"Abstract Text:\n{row[text_col]}\n\n"
                        f"INPUT CANDIDATE LIST (Contains exactly {n_candidates} pairs):\n"
                        f"{candidate_list_str}\n\n"
                        f"INSTRUCTION:\n"
                        f"Process the {n_candidates} pairs above sequentially. For each pair, map the names exactly, "
                        f"generate the anchored chain of thought, and assign the appropriate weak_label string ('0', '1', '2', or '3').\n"
                        f"Your output JSON array MUST contain exactly {n_candidates} elements matching the order of the list above."
                    )
                
                # --- BRANCH B: MICRO LOGIC (Pair-by-Pair) ---
                else:
                    current_max_tokens = 512 if max_tokens == "auto" else max_tokens
                    gen_kwargs[max_tokens_arg] = current_max_tokens
                    
                    if verbose:
                        verbose_print(
                            verbose,
                            "Micro mode detected.",
                            f"Chemical: {row['chemical']}",
                            f"Disease: {row['disease']}",
                            f"Max tokens set to: {current_max_tokens}",
                            f"Text preview: {row['masked_text'][:160]}{'...' if len(row['masked_text']) > 160 else ''}",
                        )
                    
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
                        **gen_kwargs,
                        messages=[
                            {"role": "system", "content": instruction},
                            {"role": "user", "content": content},
                        ]
                    )

                    if is_macro:
                        # Flatten out the array of items generated for this single abstract
                        # Create a set of (chemical, disease) tuples from LLM response
                        llm_pairs = {(rel.chemical, rel.disease) for rel in analysis.relationships}
                        
                        # Create rows for all LLM-returned pairs
                        for rel in analysis.relationships:
                            rel_dict = rel.model_dump()
                            rel_dict[id_col] = item_id
                            rel_dict["extraction_status"] = "SUCCESS"
                            batch_results.append(rel_dict)
                        
                        # Check for missing candidate pairs (that LLM didn't return)
                        expected_pairs = {(r['chemical'], r['disease']) for _, r in doc_candidates.iterrows()}
                        missing_pairs = expected_pairs - llm_pairs
                        
                        for chem, disease in missing_pairs:
                            batch_results.append({
                                id_col: item_id,
                                "chemical": chem,
                                "disease": disease,
                                "chain_of_thought": "SKIPPED BY MODEL: Pair was in candidate list but not evaluated",
                                "weak_label": None,
                                "extraction_status": "INCOMPLETE"
                            })
                    else:
                        # Single row processing
                        rel_dict = analysis.model_dump()
                        rel_dict[id_col] = item_id
                        rel_dict["extraction_status"] = "SUCCESS"
                        batch_results.append(rel_dict)

                except Exception as e:
                    print(f"❌ Error on row {index} ({id_col}: {item_id}): {e}")
                    
                    if is_macro:
                        for _, r in doc_candidates.iterrows():
                            batch_results.append({
                                id_col: item_id,
                                "chemical": r["chemical"],
                                "disease": r["disease"],
                                "chain_of_thought": f"FAILED GENERATION: {str(e)[:50]}",
                                "weak_label": None,
                                "extraction_status": "FAILED"
                            })
                    else:
                        batch_results.append({
                            id_col: item_id,
                            "chemical": row.get("chemical", None),
                            "disease": row.get("disease", None),
                            "chain_of_thought": f"FAILED GENERATION: {str(e)[:50]}",
                            "weak_label": None,
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
        for col in ['chemical', 'disease']:
            if col not in structure_only_df.columns:
                structure_only_df[col] = None

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