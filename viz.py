import streamlit as st
import json
from pathlib import Path

st.write("# Hi")

path = Path("output/llemma7b_minif2f_valid/13-02-2024-15-28/results__EleutherAI_llemma_7b__0.json")
with path.open("r") as f:
    output = json.load(f)

args = output["args"]
results = output["results"]

id = st.number_input("Problem number", 0, len(results)-1)
result = results[id]

example = result["example"]

attempts = result["attempt_results"]

st.write("Split", example["split"])
st.write("Problem", example["full_name"])
st.write("#", example["statement"])

st.write("Success", result["success"])

st.write("# Attempts")
attempt_id = st.number_input("Attempt number", 0, len(attempts)-1)
attempt = attempts[attempt_id]
st.write(attempt)
