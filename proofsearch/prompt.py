# https://github.com/National-Zoning-Atlas/zoning-gpt/blob/master/zoning/prompting.py
 
import openai

client = openai.OpenAI()



def _unique_sorted(texts, scores):
    texts_ = []
    scores_ = []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_, scores_


def generate(prompt, model, tokenizer, temperatures, num_samples, stop, max_tokens=256):
    output = client.chat.completions.create(
        model="gpt-4-1106-preview",
        temperature=temperatures[0],
        max_tokens=max_tokens,
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ],
        stop = stop,
        logprobs = True,
        n = num_samples,
    )
    contents = [choice.message.content for choice in output.choices]
    # normalize by logprobs by length?
    logprobs = [sum(x.logprob for x in choice.logprobs.content) for choice in output.choices]
    return _unique_sorted(contents, logprobs)


def generate_vllm(prompt, model, tokenizer, temperatures, num_samples, stop, max_tokens=256):
    texts, scores = [], []
    for temperature in temperatures:
        params = vllm.SamplingParams(
            n=num_samples,
            temperature=temperature,
            use_beam_search=temperature==0.0,
            max_tokens=max_tokens,
            stop=stop,
        )
        outputs = model.generate([prompt], params, use_tqdm=False)
        if len(outputs) == 0:
            return [], []
        for output in outputs[0].outputs:
            text = output.text.replace(tokenizer.eos_token, '')
            score = output.cumulative_logprob/max(len(output.token_ids), 1)
            texts.append(text)
            scores.append(score)

    texts, scores = _unique_sorted(texts, scores)
    return texts, scores


def _prompt_fewshot(ts):
    prompt = """Given the Lean 4 tactic state, suggest a next tactic.
Give only the tactic without explanation.

Here are some examples, be sure to follow their formatting:

Tactic state:
---
α : Type u_1
r : α → α → Prop
inst✝¹ : DecidableEq α
inst✝ : IsIrrefl α r
⊢ CutExpand r ≤ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) ↑toFinsupp
---
Next tactic:
---
rintro s t ⟨u, a, hr, he⟩
---

Tactic state:
---
ι : Type u_1
I✝ J✝ : Box ι
x y : ι → ℝ
I J : WithBot (Box ι)
⊢ ↑I = ↑J ↔ I = J
---
Next tactic:
---
simp only [Subset.antisymm_iff, ← le_antisymm_iff, withBotCoe_subset_iff]
---

Tactic state:
---
m n : ℕ
h : Nat.coprime m n
⊢ Nat.gcd m n = 1
---
Next tactic:
---
rw [← h.gcd_eq_one]
---

Tactic state:
---
%s
---
Next tactic:
---""" % (ts)
    return prompt

