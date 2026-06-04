PROMPT_TEMPLATE = """
You are an expert biomedical curator. 
Analyze the abstract text below to identify if a causal relationship exists between the listed entities.

Abstract:
{abstract_text}

Entities to Check:
Chemicals: {chemicals_list}
Diseases: {diseases_list}

Task: Return a strict JSON array of objects detailing where a chemical physically induces or causes a disease. Do not include introductory text or markdown formatting outside of valid JSON.

Format:
[
  {{"chemical_id": "MESH_D001906", "disease_id": "MESH_D006261", "evidence": "exact sentence match"}}
]
"""