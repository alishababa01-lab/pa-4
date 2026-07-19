"""Python client SDK for the deployed Document Analyst (Part 3)."""

from __future__ import annotations

import os
import time
import json
import requests
from collections.abc import Iterator


class AnalystClientError(Exception):
    def __init__(self, message: str, status_code: int | None = None, request_id: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


class DocumentAnalystClient:
    def __init__(
        self,
        endpoint_name: str,
        host: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        """Initializes the Document Analyst client library.
        
        If host or token are not explicitly provided, they are retrieved from
        the DATABRICKS_HOST and DATABRICKS_TOKEN environment variables.
        """
        self.endpoint_name = endpoint_name
        self.timeout = timeout
        self.max_retries = max_retries

        # Resolve host and clean any trailing slashes
        raw_host = host or os.environ.get("DATABRICKS_HOST", "")
        if not raw_host:
            raise ValueError(
                "Databricks host must be provided or set via the DATABRICKS_HOST environment variable."
            )
        self.host = raw_host.rstrip("/")

        # Resolve token
        self.token = token or os.environ.get("DATABRICKS_TOKEN", "")
        if not self.token:
            raise ValueError(
                "Databricks token must be provided or set via the DATABRICKS_TOKEN environment variable."
            )

        # Setup standard endpoints and headers
        self.invocations_url = f"{self.host}/serving-endpoints/{self.endpoint_name}/invocations"
        self.status_url = f"{self.host}/api/2.0/serving-endpoints/{self.endpoint_name}"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """Executes HTTP requests with exponential backoff for 429/503 and strict timeout enforcement."""
        start_time = time.time()
        retries = 0
        backoff_factor = 1.5

        while True:
            elapsed = time.time() - start_time
            remaining_timeout = self.timeout - elapsed

            if remaining_timeout <= 0:
                raise TimeoutError(f"Request timed out after {elapsed:.2f} seconds.")

            # Apply remaining timeout limit to the individual connection attempt
            kwargs["timeout"] = min(remaining_timeout, self.timeout)

            try:
                response = requests.request(method, url, **kwargs)
            except requests.exceptions.Timeout as e:
                total_elapsed = time.time() - start_time
                raise TimeoutError(f"Request timed out after {total_elapsed:.2f} seconds.") from e
            except requests.exceptions.RequestException as e:
                raise AnalystClientError(f"A network transport error occurred: {e}") from e

            # Handle retryable transient status codes (429: Rate Limit, 503: Service Unavailable/Scaling)
            if response.status_code in (429, 503):
                retries += 1
                if retries > self.max_retries:
                    req_id = response.headers.get("x-request-id")
                    raise AnalystClientError(
                        message=f"Max retries ({self.max_retries}) exceeded for server status {response.status_code}.",
                        status_code=response.status_code,
                        request_id=req_id
                    )

                # Calculate exponential backoff capped by total remaining time
                sleep_time = min(backoff_factor ** retries, self.timeout - (time.time() - start_time))
                if sleep_time <= 0:
                    raise TimeoutError(f"Request timed out while preparing to backoff-retry after {time.time() - start_time:.2f} seconds.")
                
                time.sleep(sleep_time)
                continue

            # Wrap non-OK error responses in AnalystClientError
            if not response.ok:
                req_id = response.headers.get("x-request-id")
                try:
                    # Parse Databricks specific error details if available
                    err_json = response.json()
                    req_id = req_id or err_json.get("request_id") or err_json.get("ReqId")
                except Exception:
                    pass

                raise AnalystClientError(
                    message=f"Request failed with status code {response.status_code}: {response.text}",
                    status_code=response.status_code,
                    request_id=req_id
                )

            return response

    def ask(self, question: str) -> str:
        """Sends a single question to the serving endpoint and returns the parsed answer string."""
        payload = {
            "messages": [{"role": "user", "content": question}]
        }
        
        response = self._request_with_retry("POST", self.invocations_url, headers=self.headers, json=payload)
        
        try:
            data = response.json()
        except Exception as e:
            raise AnalystClientError(f"Failed to parse JSON response from serving endpoint: {response.text}") from e

        # Resolve response parsing (Handles Path A [LangGraph List] & Path B [OpenAI ChatCompletion])
        if isinstance(data, list) and len(data) > 0:
            # Path A: LangGraph State List (mlflow.langchain.log_model)
            state = data[0]
            if isinstance(state, dict):
                if "final_answer" in state:
                    return state["final_answer"]
                if "messages" in state and len(state["messages"]) > 0:
                    last_msg = state["messages"][-1]
                    if isinstance(last_msg, dict) and "content" in last_msg:
                        return last_msg["content"]
            raise AnalystClientError(f"Invalid schema received for list-based response payload: {data}")

        elif isinstance(data, dict):
            # Path B: Standard OpenAI ChatCompletion wrapper
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
            elif "final_answer" in data:
                return data["final_answer"]
            elif "messages" in data and len(data["messages"]) > 0:
                last_msg = data["messages"][-1]
                if isinstance(last_msg, dict) and "content" in last_msg:
                    return last_msg["content"]

        raise AnalystClientError(f"Could not parse valid answer from response layout: {data}")

    def ask_streaming(self, question: str) -> Iterator[str]:
        """Queries the serving endpoint and streams the text answer chunks back as they arrive."""
        payload = {
            "messages": [{"role": "user", "content": question}],
            "stream": True
        }

        response = None  # 1. Initialize response to None to prevent UnboundLocalErrors
        
        # Request streaming flag enabled with graceful fallback for unsupported endpoints
        try:
            try:
                response = self._request_with_retry(
                    "POST", self.invocations_url, headers=self.headers, json=payload, stream=True
                )
            except AnalystClientError as e:
                # Fallback gracefully if the endpoint rejects the stream parameter (HTTP 400)
                if e.status_code == 400 and "streaming" in str(e).lower():
                    yield self.ask(question)
                    return  # Safe to return now because response is initialized to None
                raise e

            full_content_accumulated = ""
            has_yielded_chunks = False

            for line in response.iter_lines():
                if not line:
                    continue

                decoded_line = line.decode("utf-8").strip()
                if not decoded_line.startswith("data:"):
                    continue

                data_str = decoded_line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk_json = json.loads(data_str)
                except Exception:
                    # Fallback to yielding raw string if not JSON format
                    yield data_str
                    has_yielded_chunks = True
                    continue

                # Parse JSON stream chunks
                if isinstance(chunk_json, dict):
                    # Check OpenAI style: choices[0].delta.content
                    if "choices" in chunk_json and len(chunk_json["choices"]) > 0:
                        choice = chunk_json["choices"][0]
                        if "delta" in choice and "content" in choice["delta"]:
                            delta_content = choice["delta"]["content"]
                            if delta_content:
                                yield delta_content
                                has_yielded_chunks = True
                            continue
                        elif "message" in choice and "content" in choice["message"]:
                            message_content = choice["message"]["content"]
                            if message_content:
                                yield message_content
                                has_yielded_chunks = True
                            continue

                    # Check LangGraph/LangChain style dictionary chunks
                    if "messages" in chunk_json and len(chunk_json["messages"]) > 0:
                        content = chunk_json["messages"][-1].get("content", "")
                        if content:
                            full_content_accumulated = content
                    elif "final_answer" in chunk_json:
                        full_content_accumulated = chunk_json["final_answer"]

                elif isinstance(chunk_json, list) and len(chunk_json) > 0:
                    # LangGraph list chunk format
                    state = chunk_json[0]
                    if isinstance(state, dict):
                        if "final_answer" in state:
                            full_content_accumulated = state["final_answer"]
                        elif "messages" in state and len(state["messages"]) > 0:
                            last_msg = state["messages"][-1]
                            if isinstance(last_msg, dict) and "content" in last_msg:
                                full_content_accumulated = last_msg["content"]

            # Streaming fallback: yield full response once if streaming didn't output incremental deltas
            if not has_yielded_chunks and full_content_accumulated:
                yield full_content_accumulated

        except Exception as e:
            if not isinstance(e, AnalystClientError):
                raise AnalystClientError(f"Error occurred during response stream execution: {e}") from e
            raise e
        finally:
            # 2. Only close if response was successfully opened
            if response is not None:
                response.close()

    def health_check(self) -> bool:
        """Queries the Databricks Serving Endpoint metadata and returns True if it is READY."""
        try:
            response = self._request_with_retry("GET", self.status_url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                state = data.get("state", {})
                if isinstance(state, dict):
                    ready_status = state.get("ready")
                    # Ensure readiness state is active
                    if isinstance(ready_status, str) and ready_status.upper() == "READY":
                        return True
            return False
        except Exception:
            # Any connection, status error, or timeout indicates the service is unhealthy
            return False