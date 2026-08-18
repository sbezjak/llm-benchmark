"""Anthropic Message Batches adapter - the async job lane, at half price.

This is the sibling of `anthropic.py`, not a `Provider`. The interactive
`Provider.measure(prompt) -> Measurement` seam is one prompt, one call, one
streamed answer with an honest time-to-first-token. A batch is a different
shape and a different question: submit many requests as one job, poll until the
job ends, then retrieve every answer keyed by `custom_id` (results come back in
any order). So it cannot hide behind `measure` - it gets its own class and its
own runner.

What the batch lane buys, and what it costs, are both real and both go in the
writeup:

- COST: every token is billed at 50% (see `pricing.batch=True`). Same suite,
  same answers, half the money.
- LATENCY: there is no per-request latency to measure. The requests are not
  streamed, so there is no first-token event (`ttft_ms` is always None), and
  they don't return one at a time - the whole job completes together after
  minutes, not the seconds an interactive call takes. The honest number is the
  WHOLE-BATCH turnaround, and this adapter reports exactly that: every
  Measurement in a batch carries the same `latency_ms` (the job's wall-clock),
  clearly the job time and not a per-request time. That is the trade you are
  buying - half the price for minutes of turnaround.

Raw httpx like the streaming adapter, so the same respx contract-test discipline
applies (no live network in unit tests, no billed call in default runs). Every
prompt going in and every answer coming out is logged at INFO so the report
shows the whole job, same rule as every other component that calls a model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import httpx

from llm_benchmark.providers.base import Measurement

logger = logging.getLogger(__name__)

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicBatchError(RuntimeError):
    """Raised when the Batches API returns an error or an unexpected payload."""


class AnthropicBatchProvider:
    """Anthropic Message Batches adapter (Haiku and Sonnet ride one class).

    The lifecycle is three HTTP calls, mirroring the docs exactly:
      1. `submit`  -> POST /v1/messages/batches         (one job, many requests)
      2. `wait`    -> GET  /v1/messages/batches/{id}     (poll processing_status)
      3. `results` -> GET  {results_url}                 (JSONL, keyed by custom_id)

    `run` chains all three and returns a `Measurement` per custom_id plus the
    job's turnaround. Not a `Provider` subclass - `measure(prompt)` is the wrong
    shape for a submit-poll-retrieve job (see the module docstring).

    Requests are non-streamed (`max_tokens` required, no `stream` field); sampling
    params are omitted for the same reason as the streaming adapter - current-gen
    models reject non-default values with a 400.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 60.0,
        max_tokens: int = 1024,
        poll_interval: float = 5.0,
        poll_timeout: float = 3600.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required (never hardcode it; read it from the env)")
        self.model = model
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.poll_interval = poll_interval
        # A batch is quoted to finish within ~1h; the poll_timeout is the loud
        # backstop so a wedged job fails instead of polling forever.
        self.poll_timeout = poll_timeout
        # The receipt. `run()` records the provider's own batch_id and final
        # request_counts here so a caller can commit them to durable evidence -
        # a batch_id that lives only in the log is a receipt one lost `tee|tail`
        # away from gone (which is exactly how the first batch run's was lost).
        self.last_batch_id: str | None = None
        self.last_request_counts: dict | None = None

    @property
    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key, "anthropic-version": ANTHROPIC_VERSION}

    async def submit(self, prompts: dict[str, str]) -> str:
        """POST one job carrying every (custom_id -> prompt). Returns the batch id.

        `custom_id` is how results are re-associated with items - the API returns
        answers in ANY order, so position is never load-bearing (the runner keys
        every capture by custom_id, never by index)."""
        url = f"{self.base_url}/v1/messages/batches"
        requests = [
            {
                "custom_id": cid,
                "params": {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
            for cid, prompt in prompts.items()
        ]
        logger.info(
            "anthropic_batch.submit model=%s url=%s n=%d\nPROMPTS:\n%s",
            self.model,
            url,
            len(requests),
            "\n".join(f"  [{cid}] {prompt}" for cid, prompt in prompts.items()),
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, json={"requests": requests}, headers=self._headers)
                if resp.status_code >= 400:
                    raise AnthropicBatchError(
                        f"{url} returned HTTP {resp.status_code}: {resp.text}"
                    )
                obj = resp.json()
            except httpx.HTTPError as e:
                raise AnthropicBatchError(f"request to {url} failed: {e}") from e
        batch_id = obj.get("id")
        if not batch_id:
            raise AnthropicBatchError(f"batch create returned no id: {json.dumps(obj)}")
        logger.info(
            "anthropic_batch.submitted model=%s batch_id=%s status=%s",
            self.model,
            batch_id,
            obj.get("processing_status"),
        )
        return batch_id

    async def _retrieve(self, batch_id: str) -> dict:
        url = f"{self.base_url}/v1/messages/batches/{batch_id}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.get(url, headers=self._headers)
                if resp.status_code >= 400:
                    raise AnthropicBatchError(
                        f"{url} returned HTTP {resp.status_code}: {resp.text}"
                    )
                return resp.json()
            except httpx.HTTPError as e:
                raise AnthropicBatchError(f"request to {url} failed: {e}") from e

    async def wait(self, batch_id: str) -> dict:
        """Poll processing_status until the job `ended` (or poll_timeout trips).

        Returns the final batch object, whose `results_url` is populated once the
        job ends. The wall-clock across this wait IS the job turnaround the report
        shows as latency - there is no other honest latency number for a batch."""
        deadline = time.monotonic() + self.poll_timeout
        while True:
            obj = await self._retrieve(batch_id)
            status = obj.get("processing_status")
            counts = obj.get("request_counts", {})
            logger.info(
                "anthropic_batch.poll model=%s batch_id=%s status=%s counts=%s",
                self.model,
                batch_id,
                status,
                json.dumps(counts),
            )
            if status == "ended":
                return obj
            if time.monotonic() >= deadline:
                raise AnthropicBatchError(
                    f"batch {batch_id} still {status!r} after {self.poll_timeout}s poll_timeout"
                )
            await asyncio.sleep(self.poll_interval)

    async def fetch_results(self, results_url: str) -> dict[str, dict]:
        """GET the JSONL results file and key each line by custom_id.

        Each line is `{custom_id, result: {type, message?}}`; a succeeded result
        carries the full non-streamed Message (content + its own usage)."""
        if not results_url:
            raise AnthropicBatchError("batch ended with no results_url")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.get(results_url, headers=self._headers)
                if resp.status_code >= 400:
                    raise AnthropicBatchError(
                        f"{results_url} returned HTTP {resp.status_code}: {resp.text}"
                    )
                body = resp.text
            except httpx.HTTPError as e:
                raise AnthropicBatchError(f"request to {results_url} failed: {e}") from e
        results: dict[str, dict] = {}
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                raise AnthropicBatchError(f"results line is not JSON: {e}") from e
            results[entry["custom_id"]] = entry["result"]
        return results

    async def run(self, prompts: dict[str, str]) -> tuple[dict[str, Measurement], dict[str, str]]:
        """Full lifecycle: submit -> wait -> fetch. Returns (measurements, errors).

        `measurements` is one Measurement per succeeded custom_id, each carrying
        the same `latency_ms` (the whole-job turnaround) and `ttft_ms=None`.
        `errors` is custom_id -> reason for anything that didn't succeed, so a
        partial failure is surfaced loudly rather than silently dropped."""
        start = time.monotonic()
        batch_id = await self.submit(prompts)
        final = await self.wait(batch_id)
        turnaround_s = time.monotonic() - start
        # Stash the receipt before fetching results so it survives even a
        # results-fetch failure - the batch_id is the console cross-reference.
        self.last_batch_id = batch_id
        self.last_request_counts = final.get("request_counts")
        results = await self.fetch_results(final.get("results_url", ""))

        measurements: dict[str, Measurement] = {}
        errors: dict[str, str] = {}
        for cid in prompts:
            result = results.get(cid)
            if result is None:
                errors[cid] = "no result returned for this custom_id"
                continue
            rtype = result.get("type")
            if rtype != "succeeded":
                # errored / canceled / expired - keep the payload so the runner
                # can log exactly what the provider said went wrong.
                errors[cid] = f"{rtype}: {json.dumps(result.get(rtype) or result)}"
                continue
            message = result["message"]
            text = "".join(
                block.get("text", "")
                for block in message.get("content", [])
                if block.get("type") == "text"
            )
            usage = message.get("usage", {}) or {}
            logger.info(
                "anthropic_batch.result model=%s custom_id=%s\n"
                "USAGE (verbatim from the provider - the source of every token count):\n%s\n"
                "RESPONSE:\n%s",
                self.model,
                cid,
                json.dumps(usage),
                text,
            )
            measurements[cid] = Measurement(
                text=text,
                tokens_in=usage.get("input_tokens", 0),
                tokens_out=usage.get("output_tokens", 0),
                # The whole-job turnaround, shared by every item in the batch.
                # There is no per-request latency in the batch lane; this is the
                # honest number and it is the same for the whole job (see module
                # docstring). ttft is None: nothing was streamed.
                latency_ms=turnaround_s * 1000.0,
                ttft_ms=None,
                model=self.model,
            )
        logger.info(
            "anthropic_batch.done model=%s batch_id=%s turnaround_s=%.1f ok=%d err=%d",
            self.model,
            batch_id,
            turnaround_s,
            len(measurements),
            len(errors),
        )
        return measurements, errors
