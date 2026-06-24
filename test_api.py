"""Quick smoke-test for the OpenAI API connection.

Checks:
    - OPENAI_API_KEY is set in the environment
    - The model can be instantiated via get_llm()
    - A simple prompt returns a non-empty response
"""

import os
import sys

from langchain_core.messages import HumanMessage

from src.models import get_llm


def test_env():
    key = os.getenv("OPENAI_API_KEY")
    assert key, "OPENAI_API_KEY is not set in the environment"
    print(f"[OK] OPENAI_API_KEY is set ({len(key)} chars)")


def test_model_init():
    llm = get_llm()
    assert llm is not None
    print(f"[OK] Model instantiated: {llm}")


def test_simple_call():
    llm = get_llm()
    response = llm.invoke([HumanMessage(content="Reply with the single word: pong")])
    text = response.content.strip()
    assert text, "Response content is empty"
    print(f"[OK] API responded: {text!r}")


if __name__ == "__main__":
    tests = [test_env, test_model_init, test_simple_call]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
    if failed:
        print(f"\n{failed}/{len(tests)} test(s) failed.")
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} tests passed.")
