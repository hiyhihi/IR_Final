"""CLI tiện ích để gọi Teacher Server: register / evaluate / reset / result."""
from __future__ import annotations

import argparse
import json
import sys

import httpx

import lam.config as config


def _headers() -> dict[str, str]:
    return {"X-Student-ID": config.STUDENT_ID, "Content-Type": "application/json"}


def _post(paths: list[str], payload=None, timeout=30) -> httpx.Response:
    last = None
    for path in paths:
        url = f"{config.TEACHER_BASE_URL}{path}"
        r = httpx.post(url, headers=_headers(), json=payload, timeout=timeout)
        if r.status_code != 404:
            return r
        last = r
    return last  # type: ignore[return-value]


def _get(paths: list[str], timeout=30) -> httpx.Response:
    last = None
    for path in paths:
        url = f"{config.TEACHER_BASE_URL}{path}"
        r = httpx.get(url, headers=_headers(), timeout=timeout)
        if r.status_code != 404:
            return r
        last = r
    return last  # type: ignore[return-value]


def register() -> None:
    payload = {"server_url": config.STUDENT_SERVER_URL}
    r = _post(["/competition/register", "/register"], payload=payload, timeout=30)
    print(r.status_code, json.dumps(r.json(), ensure_ascii=False, indent=2))


def evaluate(document_received: bool = False) -> None:
    payload = {"document_received": document_received}
    r = _post(["/competition/evaluate", "/evaluate"], payload=payload, timeout=7200)
    print(r.status_code, json.dumps(r.json(), ensure_ascii=False, indent=2))


def reset() -> None:
    r = _post(["/competition/reset", "/reset"], timeout=30)
    print(r.status_code, json.dumps(r.json(), ensure_ascii=False, indent=2))


def result() -> None:
    r = _get(["/competition/result", "/result"], timeout=30)
    print(r.status_code, json.dumps(r.json(), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher Server client")
    parser.add_argument("action", choices=["register", "evaluate", "reset", "result"])
    parser.add_argument(
        "--doc-received",
        action="store_true",
        dest="document_received",
        help="Gửi document_received=true khi evaluate lần 2-5.",
    )
    args = parser.parse_args()

    if args.action == "evaluate":
        evaluate(document_received=args.document_received)
        return

    {
        "register": register,
        "reset": reset,
        "result": result,
    }[args.action]()

 
if __name__ == "__main__":
    try:
        main()
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        sys.exit(1)
