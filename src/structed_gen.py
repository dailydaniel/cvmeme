import sglang as sgl
from sglang.srt.constrained import build_regex_from_object


@sgl.function
def cv_parsing_gen(s, prompt, schema):
    s += prompt + "\n\n" if prompt else ""
    s += "Give me a answer in the JSON format.\n"
    s += sgl.gen(
        "json_output",
        max_tokens=512,
        temperature=0,
        regex=build_regex_from_object(schema),
    )


def driver_cv_parsing_gen(prompt: str, schema):
    state = cv_parsing_gen.run(prompt, schema)
    return state.text()
