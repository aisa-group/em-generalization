from typing import Dict, List, Optional
import math
from functools import lru_cache
from pathlib import Path
import yaml

from openai import AsyncOpenAI


class OpenAiJudge:
    """OpenAI models tokenize all numbers from 0-100 as single tokens, which is why we can get exactly 
    one completion token with logprobs. Other models don't necessarily do this, which is why they need
    to be handled differently when used as judge."""

    def __init__(self, model: str, prompt_template: str):
        self.model = model
        self.prompt_template = prompt_template
        self.client = AsyncOpenAI()

    async def judge(self, **kwargs):
        messages = [dict(role='user', content=self.prompt_template.format(**kwargs))]
        logprobs = await self.logprob_probs(messages)
        # gpt-5 need to do 0-10
        if "gpt-5" in self.model:
            score_scale = 10
        else:
            score_scale = 100
        score = self._aggregate_0_100_score(logprobs, score_scale=score_scale)
        return score

    async def judge_text(self, **kwargs):
        messages = [
            {
                "role": "user",
                "content": self.prompt_template.format(**kwargs)
            }
        ]

        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=1,
            seed=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "score_output",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100
                            }
                        },
                        "required": ["score"],
                        "additionalProperties": False
                    }
                }
            }
        )

        try:
            content = completion.choices[0].message.content
            import json
            data = json.loads(content)

            score = data.get("score")
            if isinstance(score, int) and 0 <= score <= 100:
                return score

        except Exception:
            pass

        return None

    async def logprob_probs(self, messages) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token."""
        completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1,
                temperature=0,
                logprobs=True,
                top_logprobs=20,
                seed=0
        )
        try:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except (IndexError, AttributeError):
            print(f"[Warning] No logprobs. finish_reason={completion.choices[0].finish_reason!r}, "
                  f"refusal={completion.choices[0].message.refusal!r}")
            return {}

        result = {}
        for el in logprobs:
            result[el.token] = float(math.exp(el.logprob))

        return result

    @staticmethod
    def _aggregate_0_100_score(score: dict, score_scale: float = 100) -> Optional[float]:
        #   NOTE: we don't check for refusals explicitly. Instead we assume that
        #   if there's at least 0.25 total weight on numbers, it's not a refusal.
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > score_scale:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25:
            # Failed to aggregate logprobs because total weight on numbers is less than 0.25.
            return None
        return sum_ / total

    async def __call__(self, **kwargs):
        if "gpt-5" in self.model:
            return await self.judge_text(**kwargs)
        else:
            return await self.judge(**kwargs)

