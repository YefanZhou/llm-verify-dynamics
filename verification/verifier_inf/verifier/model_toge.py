import time
import os
from typing import List, Optional
from openai import AsyncOpenAI
from openai._exceptions import (
    RateLimitError,
    APITimeoutError,
    APIError,
)
import asyncio

TOGETHER_API_KEY = ""
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

OPENAI_API_KEY = ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


MAX_CONCURRENCY  = 2048          # raise/lower until you rarely see 429s
MAX_RETRIES      = 60
INITIAL_BACKOFF  = 2          # seconds
TIMEOUT_SECONDS  = 140


class TogetherAIInference:
    def __init__(
        self,
        model: str = '',
        provider: str = 'together',
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        **kwargs
    ):
        self.model = model
        self.provider = provider
        self.max_concurrency = MAX_CONCURRENCY
        self.max_retries = MAX_RETRIES
        self.initial_backoff = INITIAL_BACKOFF
        self.timeout_seconds = TIMEOUT_SECONDS
        
        # Sampling parameters
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.n = n
        
        # Store any additional kwargs for future use
        self.additional_params = kwargs
        
        if self.provider == 'together':
            self.api_key = TOGETHER_API_KEY
            self.base_url = "https://api.together.xyz/v1"
            os.environ["TOGETHER_API_KEY"] = self.api_key
            # TogetherAI can handle high concurrency
            self.max_concurrency = min(self.max_concurrency, 100)

        elif self.provider == 'openai':
            self.api_key = OPENAI_API_KEY
            self.base_url = "https://api.openai.com/v1"
            # OpenAI rate limits vary by model and tier
            if 'gpt-4o' in model.lower():  # This covers both gpt-4o and gpt-4o-mini, but gpt-4o-mini is caught above
                self.max_concurrency = min(self.max_concurrency, 100)  # 500 RPM / 5 = ~100 concurrent
            else:
                self.max_concurrency = min(self.max_concurrency, 50)   # Conservative default
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'together' or 'openai'")


        # Log the sampling parameters
        print("=== APIInference Initialized ===")
        print(f"Provider: {self.provider}")
        print(f"Base URL: {self.base_url}")
        print(f"Model: {self.model}")
        print(f"Max Concurrency: {self.max_concurrency}")
        print(f"Max Retries: {self.max_retries}")
        print(f"Timeout: {self.timeout_seconds}s")
        print("Sampling Parameters:")
        print(f"  Temperature: {self.temperature}")
        print(f"  Top-p: {self.top_p}")
        print(f"  Max Tokens: {self.max_tokens}")
        print(f"  N (sequences): {self.n}")
        if self.additional_params:
            print(f"  Additional Params: {self.additional_params}")
        print("=" * 40)

    def show_sampling_params(self):
        """Display current sampling parameters for verification."""
        print("Current Sampling Parameters:")
        print(f"  Temperature: {self.temperature}")
        print(f"  Top-p: {self.top_p}")
        print(f"  Max Tokens: {self.max_tokens}")
        print(f"  N (sequences): {self.n}")
        if self.additional_params:
            print(f"  Additional Params: {self.additional_params}")

    def _init_client(self) -> AsyncOpenAI:
        """
        Initialise an async OpenAI-style client that talks to Together.
        Expects your Together key in $TOGETHER_API_KEY
        """
        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    async def _single_call(
        self,
        client: AsyncOpenAI,
        message: list,
    ) -> List[str]:
        """
        One chat completion with retries & exponential back-off.
        Returns the assistant message content(s).
        """
        backoff = self.initial_backoff
        for attempt in range(1, self.max_retries + 1):
            try:
                # Build request parameters
                request_params = {
                    "model": self.model,
                    "messages": message, #[{"role": "user", "content": prompt}],
                    "timeout": self.timeout_seconds,
                }
                
                # Add sampling parameters if provided
                if self.temperature is not None:
                    request_params["temperature"] = self.temperature
                if self.top_p is not None:
                    request_params["top_p"] = self.top_p
                if self.max_tokens is not None:
                    request_params["max_tokens"] = self.max_tokens
                if self.n is not None:
                    request_params["n"] = self.n
                
                if self.model == "Qwen/Qwen3-235B-A22B-Instruct-2507-tput":
                    #print('set top_k to 20')
                    request_params["extra_body"] = {'top_k': 20}
                    #request_params["extra_body"] = {
                    #            "chat_template_kwargs": {"enable_thinking": False}
                    #        }
                # Add any additional parameters
                request_params.update(self.additional_params)
                
                # Log the parameters being used (only on first attempt to avoid spam)
                #if attempt > 3:
                #    print(f"Making API call with params: {request_params}")
                
                resp = await client.chat.completions.create(**request_params)

                # Return all responses if n > 1, otherwise just the first one
                if self.n and self.n > 1:
                    return [choice.message.content for choice in resp.choices]
                else:
                    #return [resp.choices[0].message.content]
                    return [resp.choices[0].message.content] if (resp.choices and len(resp.choices) > 0) else ["API_ERROR: No choices in response"]

            except RateLimitError:
                # Too many requests / tokens – wait, then try again.
                await asyncio.sleep(backoff)
                backoff *= 2

            except APITimeoutError:
                # Transient network/server issue - continue retrying
                await asyncio.sleep(backoff)
                backoff *= 2

            except APIError as e:
                # Transient network/server issue.
                error_str = str(e)
                # Check for non-retryable errors that should return immediately , '401', '403', '404'
                if any(code in error_str for code in ['Error code: 422']) and \
                        any(msg in error_str.lower() for msg in ['input validation error', 'invalid_request_error', 'context length', 'token limit']):
                    
                    error_msg = f"API_ERROR: {error_str}"
                    print(f"Non-retryable error: {error_msg}")
                    if self.n and self.n > 1:
                        return [error_msg] * self.n
                    else:
                        return [error_msg]
                
                # Retryable API error - continue retrying
                print(f"Retryable API Error (attempt {attempt}/{self.max_retries}): {error_str}")
                await asyncio.sleep(backoff)
                backoff *= 2

        raise RuntimeError(f"Gave up after {self.max_retries} attempts for: {message!r}")

    async def generate(self, messages: List[list]) -> List[List[str]]:
        """
        Generate responses for a list of prompts using Together AI.
        
        Args:
            prompts: List of prompt strings to send to the model
            
        Returns:
            List of response lists from the model (each prompt can have multiple responses if n > 1)
        """
        start_time = time.time()
        client = self._init_client()
        semaphore = asyncio.Semaphore(self.max_concurrency)
        completed = 0
        total = len(messages)

        async def bounded_call(p: list, idx: int) -> List[str]:
            nonlocal completed
            async with semaphore:
                try:
                    result = await self._single_call(client, p)
                    completed += 1
                    #print(f"✓ Completed {completed}/{total} requests ({(completed/total)*100:.1f}%)")
                    elapsed = time.time() - start_time if 'start_time' in locals() else 0
                    if completed > 5 and completed % 10==0:  # Only show ETA after a few completions for accuracy
                        avg_time = elapsed / completed
                        remaining_time = (total - completed) * avg_time
                        eta_min = remaining_time / 60
                        print(f"✓ Completed {completed}/{total} requests ({(completed/total)*100:.1f}%) | ETA: {eta_min:.1f}m")
                    else:
                        print(f"✓ Completed {completed}/{total} requests ({(completed/total)*100:.1f}%)")
                    return result
                
                except Exception as e:
                    completed += 1
                    print(f"✗ Failed {completed}/{total} requests - {e}")
                    raise

        tasks = [asyncio.create_task(bounded_call(p, i)) for i, p in enumerate(messages)]
        return await asyncio.gather(*tasks)

