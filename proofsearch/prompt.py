# https://github.com/National-Zoning-Atlas/zoning-gpt/blob/master/zoning/prompting.py
 
import openai

client = openai.OpenAI()


async def prompt(
    model_name: str,
    input_prompt: str | list[dict[str, str]],
    max_tokens=256,
    formatted_response=False,
    temperature=0.0,
) -> str | None:
    base_params = {
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    match model_name:
        case "text-davinci-003":
            resp = await client.completions.create(
                **base_params,
                prompt=input_prompt,
            )
            top_choice = resp.choices[0]  # type: ignore
            return top_choice.text
        case "gpt-3.5-turbo" | "gpt-4":
            resp = await client.chat.completions.create(
                **base_params,
                messages=input_prompt,
            )
            top_choice = resp.choices[0]  # type: ignore
            return top_choice.message.content
        case "gpt-4-1106-preview":
            if not formatted_response:
                resp = await client.chat.completions.create(
                    **base_params,
                    messages=input_prompt,
                )
                top_choice = resp.choices[0]  # type: ignore
                return top_choice.message.content
            else:
                resp = await client.chat.completions.create(
                    **base_params,
                    messages=input_prompt,
                    response_format={"type": "json_object"}
                )
                top_choice = resp.choices[0]  # type: ignore
                return top_choice.message.content
        case _:
            raise ValueError(f"Unknown model name: {model_name}")
