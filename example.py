import os
import json
import logging
import typing
from pathlib import Path

import pandas as pd
from openai import OpenAI
from sqlalchemy import create_engine, text
from pydantic import BaseModel, Field

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filename="stage1_reworked.log",
)
logger = logging.getLogger("stage1")

stage1 = Path("stage1").resolve()
stage1.mkdir(exist_ok=True)

engine = create_engine(os.environ["DATABASE"])

client = OpenAI(api_key=os.environ["OPENAI"])
QUERY = """\
SELECT
    taxonomy.id AS "taxonomy_id",
    document_section_chunk.id AS "chunk_id",
    taxonomy.taxonomy_reasoning,
    taxonomy.chunk_level_reasoning,
    document_section_chunk.content AS chunk_text
FROM
    taxonomy
LEFT JOIN
    document_section_chunk
    ON
    taxonomy.chunk_id = document_section_chunk.id
LEFT JOIN
    document_section
    ON
    document_section_chunk.document_section_id = document_section.id
LEFT JOIN
    uploaded_document
    ON
    document_section.uploaded_document_id = uploaded_document.id
WHERE
    uploaded_document.language = 'RU';
"""

system_prompt = """\
You are a NATO analyst. Your core task is to determine whether a text chunk provides evidence or indications about Russia’s future ground-focused threat toward NATO.

# Output Format
- Each response contains:
  - chunk_id: the unique identifier for the text chunk
  - label: 1 if the corresponding text meets the criteria, 0 if it does not

Input:\n
"""

class Response(BaseModel):
    id: str = Field(..., description="The unique identifier for the text chunk.")
    label: typing.Literal["1", "0"] = Field(
        ..., description="A single integer: 1 if the text meets the criteria, 0 if it does not."
    )


if __name__ == "__main__":
    data = pd.read_sql(text(QUERY), engine)

    data["input"] = (
        "Chunk Text: " + data["chunk_text"] + "\n" +
        "Document Context: " + data["taxonomy_reasoning"] + "\n" + data["chunk_level_reasoning"]
    )

    for item in data.to_dict(orient="records"):
        user_content = json.dumps({"id": item["taxonomy_id"], "text": item["text"]})
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=100,
            response_format=Response
        )
        with open(f"{item['taxonomy_id']}.json", "w", encoding="utf-8") as f:
            f.write(response.to_json())
