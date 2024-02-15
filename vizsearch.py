# Lean proof search with LeanDojo interaction
# Author: Sean Welleck
import json
import heapq
import subprocess
import time
import transformers
import vllm
from datetime import datetime
from lean_dojo import *
from pathlib import Path
from tqdm import tqdm, trange
import asyncio

from proofsearch.prompt import client


os.environ['TOKENIZERS_PARALLELISM'] = 'false'


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


def _unique_sorted(texts, scores):
    texts_ = []
    scores_ = []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_, scores_


def _tactic_state(state):
    if isinstance(state, TacticState):
        ts = state.pp
    else:
        ts = state.unsolved_tactic_state
    return ts


def _prompt_fewshot(ts):
    prompt = """Given the Lean 4 tactic state, suggest a next tactic.
Here are some examples:

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


def best_first_search(
        theorem,
        model,
        tokenizer,
        max_iters,
        temperatures,
        num_samples,
        prompt_fn,
        timeout=600,
        early_stop=False,
        max_tokens=256
) -> dict:
    """Best first search."""
    attempt_results = []
    import pdb; pdb.set_trace()
    try:
        with Dojo(theorem, hard_timeout=timeout) as (dojo, init_state):
            start = time.time()
            proof_finished = False
            queue = [(0.0, [], init_state, [])]
            visited = set()

            for iteration in trange(max_iters):
                if len(queue) == 0 or proof_finished:
                    break

                total_score, steps, state, trace = heapq.heappop(queue)
                ts = _tactic_state(state)
                visited.add(ts)

                #step_cands, step_scores = generate_vllm(
                step_cands, step_scores = generate(
                    prompt_fn(ts),
                    model,
                    tokenizer,
                    temperatures,
                    num_samples,
                    stop='---',
                    max_tokens=max_tokens
                )
                step_cands = [s.strip() for s in step_cands]

                print(step_cands)

                for step, score in zip(step_cands, step_scores):
                    result = dojo.run_tac(state, step)
                    step_trace = {
                        "tactic": step,
                        "state_before": _tactic_state(state)
                    }
                    if isinstance(result, ProofFinished):
                        attempt_results.append({
                            'theorem': theorem.full_name,
                            'proof': steps + [step],
                            'score': total_score - score,
                            'success': True,
                            'failure_reason': '',
                            'trace': trace + [step_trace],
                            'temperature': temperatures,
                            'elapsed': start - time.time(),
                            'iteration': iteration
                        })
                        if early_stop:
                            return attempt_results
                        proof_finished = True
                        break
                    elif isinstance(result, TacticState):
                        if _tactic_state(result) not in visited:
                            # Score is negative log probability summed across steps
                            new_score = (total_score - score)
                            heapq.heappush(
                                queue, (new_score, steps+[step], result, trace+[step_trace])
                            )
    except (DojoInitError, DojoHardTimeoutError, DojoCrashError, subprocess.CalledProcessError) as e:
        if len(attempt_results) == 0:
            attempt_results.append({
                'theorem': theorem.full_name,
                'success': False,
                'failure_reason': type(e).__name__
            })

    if len(attempt_results) == 0:
        attempt_results.append({
            'theorem': theorem.full_name,
            'success': False,
            'failure_reason': 'SearchEnded'
        })

    return attempt_results

def _load_model(tp_degree):
    QUANTIZED = False
    if QUANTIZED:
        # yikes, it looks like awq is really bad?
        model_name = "TheBloke/llemma_7b-AWQ"
        model = vllm.LLM(
            model=model_name,
            quantization = "awq",
            tensor_parallel_size=tp_degree,
            #dtype='bfloat16',
            #dtype='float16',
            dtype = "auto",
            max_num_batched_tokens=4096
        )
    else:
        model_name = "EleutherAI/llemma_7b"
        model = vllm.LLM(
            model=model_name,
            #quantization = "awq",
            tensor_parallel_size=tp_degree,
            #dtype='bfloat16',
            #dtype='float16',
            dtype = "auto",
            max_num_batched_tokens=4096
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def _load_data(dataset_name, dataset_path):
    if 'minif2f' in dataset_name:
        data = []
        with open(dataset_path) as f:
            for line in f.readlines():
                data_ = json.loads(line)
                assert data_['commit'] == 'd00c776260c77de7e70125ef0cd119de6c0ff1de'
                data.append(data_)

        if 'valid' in dataset_name:
            data = [x for x in data if x['split'] == 'valid']
        else:
            data = [x for x in data if x['split'] == 'test']
        repo = LeanGitRepo(data[0]['url'], data[0]['commit'])
    else:
        raise NotImplementedError(dataset_name)

    return repo, data


def print_stats(results):
    print(len([x for x in results if x['success']]) / len(results))
    print("# successes: ", len([x for x in results if x['success']]), sep="\t")


def make_output_dir(output_dir):
    dt = datetime.now().strftime("%d-%m-%Y-%H-%M")
    output_dir = os.path.join(output_dir, dt)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-name',
        default='minif2f-valid',
        choices=['minif2f-valid', 'minif2f-test']
    )
    parser.add_argument('--dataset-path', default='data/minif2f.jsonl')
    parser.add_argument('--tp-degree', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--num-samples', type=int, default=32)
    parser.add_argument('--max-iters', type=int, default=100)
    parser.add_argument('--clear-process-hours', type=int, default=3)
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.0])
    args = parser.parse_args()

    repo, data = _load_data(args.dataset_name, args.dataset_path)
    print("Num examples: %d" % (len(data)))

    results = []

    #model, tokenizer = _load_model(args.tp_degree)
    model, tokenizer = None, None

    start = time.time()
    for example in tqdm(data, total=len(data)):
        file_path = example['file_path']
        theorem_name = example['full_name']
        if theorem_name not in ["mathd_numbertheory_466"]:
            continue
        theorem = Theorem(repo, file_path, theorem_name)
        attempt_results = best_first_search(
            theorem, model, tokenizer,
            max_iters=args.max_iters,
            prompt_fn=_prompt_fewshot,
            temperatures=args.temperatures,
            num_samples=args.num_samples,
            timeout=args.timeout,
            early_stop=args.early_stop
        )
        result = {
            'attempt_results': attempt_results,
            'success': any([x['success'] for x in attempt_results]),
            'example': example
        }
        results.append(result)

        print_stats(results)
        # The proof search occasionally leaves Lean processes open. As a workaround,
        # we periodically kill all Lean processes. Note that this may cause a proof search failure.
        hours = 60*60*args.clear_process_hours
        if time.time() - start > hours:
            print("=== Killing active leanprover processes to mitigate leak")
            os.system("ps aux | grep leanprover | awk '{print $2}' | xargs kill -9")
            start = time.time()

