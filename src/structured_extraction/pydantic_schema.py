from pydantic import BaseModel, Field
from typing import Literal

class RelationExtraction(BaseModel):
    chain_of_thought: str = Field(
        description=(
            "A brief 1-sentence analysis tracing the direct interaction between the tagged chemical and tagged disease. "
            )
    )
    weak_label: Literal[0, 1, 2, 3] = Field(
        description=(
            "1 if the chemical causes/induces the disease; "
            "2 if the chemical treats/prevents the disease; "
            "3 if the chemical blocks/reverses/antagonizes a drug-induced state; "
            "0 if there is no direct link, an explicit negative relation, or a non-disease polysemy occurrence."
        )
    )
