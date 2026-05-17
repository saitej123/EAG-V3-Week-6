from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import asyncio
import threading
from loguru import logger
import sys
from dotenv import load_dotenv

from llm_env import agent_run_max_seconds

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

_templates_dir = BASE_DIR / "templates"

templates = Jinja2Templates(directory=str(_templates_dir))

_agent_lock = threading.Lock()
_cognitive_agent = None


def _get_cognitive_agent():
    """Load agent (GenAI + MCP) only when needed — faster uvicorn/FastAPI startup and first page load."""
    global _cognitive_agent
    if _cognitive_agent is not None:
        return _cognitive_agent
    with _agent_lock:
        if _cognitive_agent is None:
            from agent6 import CognitiveAgent

            _cognitive_agent = CognitiveAgent()
        return _cognitive_agent

# SSE queue + thread-safe bridge (Loguru may call sinks from worker threads when enqueue=True elsewhere,
# or from sync code paths — asyncio.Queue is not thread-safe without scheduling onto the main loop).
log_queue: asyncio.Queue[str] = asyncio.Queue()
_app_loop_holder: dict[str, asyncio.AbstractEventLoop | None] = {"loop": None}


class QueueSink:
    """Send formatted log lines to the SSE queue from any thread."""

    def write(self, message: str) -> None:
        text = message.rstrip("\r\n")
        if not text:
            return
        loop = _app_loop_holder["loop"]
        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(log_queue.put_nowait, text)
            except RuntimeError:
                pass
            return
        try:
            asyncio.get_running_loop().call_soon_threadsafe(log_queue.put_nowait, text)
        except RuntimeError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    _app_loop_holder["loop"] = asyncio.get_running_loop()
    yield
    _app_loop_holder["loop"] = None


app = FastAPI(title="E-Commerce Price Analysis API", lifespan=lifespan)

# Log sinks: console + UI stream (ANSI colors for browser via ansi_up on /stream-logs)
_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)
logger.remove()
logger.add(
    sys.stdout,
    format=_LOG_FORMAT,
    colorize=sys.stdout.isatty(),
)
logger.add(
    QueueSink(),
    format=_LOG_FORMAT,
    colorize=True,
)

if not _templates_dir.is_dir():
    logger.warning(f"Templates directory not found at {_templates_dir}; GET / may fail.")

class QueryRequest(BaseModel):
    query: str


# Only one agent run at a time so logs/SQLite/memory updates do not interleave in the UI.
_run_guard: dict[str, bool] = {"busy": False}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/run-agent")
async def run_agent(request: QueryRequest):
    if _run_guard["busy"]:
        return JSONResponse(
            {"status": "busy", "detail": "An agent run is already in progress. Wait for it to finish."},
            status_code=429,
        )
    _run_guard["busy"] = True
    logger.info(f"[UI] Starting agent with query: {request.query}")

    async def _job():
        try:
            await asyncio.wait_for(
                _get_cognitive_agent().run(request.query),
                timeout=agent_run_max_seconds(),
            )
        except asyncio.TimeoutError:
            logger.error(
                f"[agent] Global time budget exceeded ({agent_run_max_seconds()}s) — run stopped"
            )
            try:
                from agent6 import _log_final_answer

                _log_final_answer(
                    "## Run stopped (time budget)\n\n"
                    "This run exceeded the configured wall-clock limit (`AGENT_RUN_MAX_SECONDS`). "
                    "See **Live console** for partial progress; narrow the query or raise the limit in `.env`."
                )
            except Exception:
                pass
            logger.info("[agent] RUN_COMPLETE reason=global_timeout")
        except asyncio.CancelledError:
            raise
        except BaseException as e:
            logger.opt(exception=e).error("[agent] Run failed")
            try:
                from agent6 import _log_final_answer

                _log_final_answer(
                    "## Run failed\n\n"
                    "An unexpected error stopped the agent. See **Live console** for the traceback."
                )
            except Exception:
                pass
            logger.info("[agent] RUN_COMPLETE reason=run_failed")
        finally:
            _run_guard["busy"] = False

    asyncio.create_task(_job())
    return {"status": "Agent started"}


@app.get("/stream-logs")
async def stream_logs():
    sse_headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    async def log_generator():
        ping_every = 12.0
        while True:
            try:
                message = await asyncio.wait_for(log_queue.get(), timeout=ping_every)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            lines = message.split("\n")
            chunk = "".join(f"data: {line}\n" for line in lines) + "\n"
            yield chunk

    return StreamingResponse(
        log_generator(),
        media_type="text/event-stream",
        headers=sse_headers,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
