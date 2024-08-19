"""
Usage:
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
python json_decode.py
"""

from enum import Enum

from pydantic import BaseModel

import sglang as sgl
from sglang.srt.constrained import build_regex_from_object

from dotenv import load_dotenv
import os
load_dotenv()


class Weapon(str, Enum):
    sword = "sword"
    axe = "axe"
    mace = "mace"
    spear = "spear"
    bow = "bow"
    crossbow = "crossbow"


class Wizard(BaseModel):
    name: str
    age: int
    weapon: Weapon


@sgl.function
def pydantic_wizard_gen(s):
    s += "Give me a description about a wizard in the JSON format.\n"
    s += sgl.gen(
        "character",
        max_tokens=128,
        temperature=0,
        regex=build_regex_from_object(Wizard),  # Requires pydantic >= 2.0
    )


def driver_pydantic_wizard_gen():
    state = pydantic_wizard_gen.run()
    print(state.text())


if __name__ == "__main__":
    sgl.set_default_backend(sgl.RuntimeEndpoint(os.environ['LLM_HOST'][:-3]))
    driver_pydantic_wizard_gen()