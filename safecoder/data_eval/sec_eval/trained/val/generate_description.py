# for each cwe in [022, 078, 079, 089], create a description for the CWE by calling the gpt-4o-mini model

import json
import logging
import os
import time

import openai


openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

def call_gpt(
    query_msg,
    openai_model_name="gpt-4o-mini",
    temp=0.7,
    max_token=128,
):
    api_call_success = False
    while not api_call_success:
        try:
            outputs = client.chat.completions.create(
                model=openai_model_name,
                messages=query_msg,
                temperature=temp,
                max_tokens=max_token,
                seed=42,
            )
            api_call_success = True
        except BaseException:
            logging.exception("An exception was thrown!")
            print("wait")
            time.sleep(2)
    assert len(outputs.choices) == 1, "API returned more than one response"
    try:
        result = outputs.choices[0].message.content
    except:
        result = outputs.choices[0].text
    return result



for cwe in["022", "078", "079", "089"]:
    # under cwe-{cwe}, there are subdirectories named as X-py where X is a number
    for subdir in os.listdir(f"cwe-{cwe}"):
        python_path = f"cwe-{cwe}/{subdir}/complete.py"
        json_path = f"cwe-{cwe}/{subdir}/info.json"
        if not subdir.endswith("-py"):
            continue

        # description_generated
        prompt = f"In one sentence, create a description for the following file (e.g., Implement a web service endpoint that executes the \"ls\" command on a specified directory and returns the result.):"

        with open(python_path) as f1, open(json_path) as f2:
            text = f1.read()
            info = json.load(f2)

        query_msg = [
            {"role": "user", "content": f"{prompt}\n\n```\n{text}\n```"}
        ]
        result = call_gpt(query_msg)
        info["description_generated"] = result.strip()

        # description
        prompt = (
            f"Summarize the following description in a few words. Give me the final short version only without any symbols. Below is an example:\n"
            f"DESCRIPTION: Implement a web service endpoint that executes the \"ls\" command on a specified directory and returns the result.\n"
            f"SHORT: use ``ls'' on a directory\n\n"
            f"DESCRIPTION: {result}\n"
            f"SHORT: "
        )
        query_msg = [
            {"role": "user", "content": prompt}
        ]
        result = call_gpt(query_msg)
        info["description"] = result.strip()

        with open(json_path, "w") as f:
            json.dump(info, f)
