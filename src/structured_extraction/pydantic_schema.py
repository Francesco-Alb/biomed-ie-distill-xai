from pydantic import BaseModel, Field
from typing import List, Literal


# --- Micro / Pair-by-Pair ---

class PairwiseExtraction(BaseModel):
    chain_of_thought: str = Field(
        description=(
            "A brief 1-sentence clinical analysis tracing the structural or therapeutic "
            "interaction between the chemical and disease based strictly on the text. "
        )
    )
    weak_label: Literal["0", "1", "2", "3"] = Field(
            description=(
                "'1' if the chemical causes/induces the disease; "
                "'2' if the chemical treats/prevents the disease; "
                "'3' if the chemical blocks/reverses/antagonizes a drug-induced state; "
                "'0' if there is no direct link, an explicit negative relation, or a non-disease polysemy occurrence."
            )
        )
    
# --- Macro / Abstract-Level ---
class MacroRelation(BaseModel):
    chemical: str = Field(description="The exact chemical name from the candidate list evaluated.")
    disease: str = Field(description="The exact disease name from the candidate list evaluated.")
    chain_of_thought: str = Field(
        description=(
            "A brief 1-sentence clinical analysis tracing the structural or therapeutic "
            "interaction between the chemical and disease based strictly on the text. "
        )
    )
    weak_label: Literal["0", "1", "2", "3"] = Field(
            description=(
                "'1' if the chemical causes/induces the disease; "
                "'2' if the chemical treats/prevents the disease; "
                "'3' if the chemical blocks/reverses/antagonizes a drug-induced state; "
                "'0' if there is no direct link, an explicit negative relation, or a non-disease polysemy occurrence."
            )
        )

class GlobalExtraction(BaseModel):
    relationships: List[MacroRelation] = Field(description="List of all candidate entity evaluations.")

