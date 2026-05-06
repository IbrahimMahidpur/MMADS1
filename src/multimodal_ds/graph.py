"""
Top-level LangGraph StateGraph — wires all agents as nodes with a
MemorySaver checkpointer for session persistence.

Fixes applied (vs original):
  1. _decide_ingestion_path: returns a single string, not a list.
     Fan-out to multiple ingestion nodes requires Send() — this simpler
     approach routes to the FIRST matching type, which is correct for
     the current sequential graph topology.
  2. _reviewer_node: task_result dict now uses keys that evaluation_agent
     actually reads ("name", "success", "output_preview", "files_created").
  3. retry_count: incremented in state when retrying, preventing infinite loops.
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def _sanitize_for_checkpoint(data):
    import numpy as np
    if isinstance(data, dict):
        return {k: _sanitize_for_checkpoint(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_sanitize_for_checkpoint(v) for v in data]
    if hasattr(data, "item") and not isinstance(data, (str, bytes)):
        return data.item()
    if isinstance(data, (np.integer, np.floating)):
        return float(data) if isinstance(data, np.floating) else int(data)
    return data


# ── Node functions ───────────────────────────────────────────────────────────

def _router_node(state):
    from pathlib import Path
    EXTENSIONS = {
        "doc":   {".pdf", ".docx", ".txt", ".md", ".html", ".rst"},
        "image": {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"},
        "audio": {".mp3", ".wav", ".m4a", ".ogg", ".flac"},
        "table": {".csv", ".xlsx", ".parquet", ".json", ".tsv"},
    }
    flags = {k: False for k in EXTENSIONS}
    for path in state.get("uploaded_files", []):
        ext = Path(path).suffix.lower()
        for kind, exts in EXTENSIONS.items():
            if ext in exts:
                flags[kind] = True
    logger.info(f"[Graph/Router] Routing flags: {flags}")
    return {"_routing_flags": flags}


def _doc_ingest_node(state):
    from multimodal_ds.ingestion.pdf_ingestion import ingest_pdf
    from multimodal_ds.ingestion.router import _ingest_plain_text
    from pathlib import Path

    DOC_EXTS = {".pdf", ".docx", ".txt", ".md", ".html", ".rst"}
    docs = list(state.get("parsed_documents", []))

    for fp in state.get("uploaded_files", []):
        if Path(fp).suffix.lower() in DOC_EXTS:
            doc = ingest_pdf(fp) if fp.endswith(".pdf") else _ingest_plain_text(fp)
            docs.append(doc.to_dict())

    vector_store_id = state.get("vector_store_id", "")
    text_chunks = [d.get("text_content", "")[:2000] for d in docs if d.get("text_content")]
    if text_chunks:
        try:
            from multimodal_ds.memory.agent_memory import AgentMemory
            mem = AgentMemory(collection_name="doc_chunks")
            for chunk in text_chunks:
                mem.store(chunk, metadata={"type": "document"})
            vector_store_id = str(mem._collection.name) if mem._collection else vector_store_id
        except Exception as e:
            logger.warning(f"[Graph/DocIngest] ChromaDB store failed: {e}")

    return {"parsed_documents": docs, "vector_store_id": vector_store_id}


def _img_ingest_node(state):
    from multimodal_ds.ingestion.image_ingestion import ingest_image, SUPPORTED_IMAGES
    from pathlib import Path

    embeddings = list(state.get("image_embeddings", []))
    for fp in state.get("uploaded_files", []):
        if Path(fp).suffix.lower() in SUPPORTED_IMAGES:
            doc = ingest_image(fp)
            if doc.embeddings:
                embeddings.append(doc.embeddings)
    return {"image_embeddings": embeddings}


def _audio_ingest_node(state):
    from multimodal_ds.ingestion.audio_ingestion import ingest_audio, SUPPORTED_AUDIO
    from pathlib import Path

    transcripts = list(state.get("audio_transcripts", []))
    for fp in state.get("uploaded_files", []):
        if Path(fp).suffix.lower() in SUPPORTED_AUDIO:
            doc = ingest_audio(fp)
            if doc.text_content:
                transcripts.append(doc.text_content)
    return {"audio_transcripts": transcripts}


def _tab_ingest_node(state):
    from multimodal_ds.ingestion.tabular_ingestion import ingest_tabular, SUPPORTED_TABULAR
    from pathlib import Path

    summaries = list(state.get("tabular_summaries", []))
    for fp in state.get("uploaded_files", []):
        if Path(fp).suffix.lower() in SUPPORTED_TABULAR:
            doc = ingest_tabular(fp)
            if doc.schema_info:
                summaries.append({
                    "source":       fp,
                    "shape":        doc.schema_info.get("shape", []),
                    "columns":      doc.schema_info.get("columns", []),
                    "dtypes":       doc.schema_info.get("dtypes", {}),
                    "sample":       doc.text_content[:1500],
                    "data_profile": doc.data_profile,
                })
    return {"tabular_summaries": _sanitize_for_checkpoint(summaries)}


def _stats_validation_node(state):
    from multimodal_ds.agents.statistical_agent import StatisticalReasoningAgent
    import pandas as pd

    uploaded = state.get("uploaded_files", [])
    tab_file = next((f for f in uploaded if f.endswith((".csv", ".xlsx", ".parquet"))), None)
    if not tab_file:
        return state

    try:
        df = pd.read_csv(tab_file) if tab_file.endswith(".csv") else pd.read_excel(tab_file)
        agent = StatisticalReasoningAgent(session_id=state.get("session_id", "default"))
        report = agent.validate_dataset(df)
        return {"statistical_report": _sanitize_for_checkpoint(report)}
    except Exception as e:
        logger.warning(f"[Graph/Stats] Validation failed: {e}")
        return {}


def _planner_node(state):
    from multimodal_ds.agents.planner_agent import run_planner
    from multimodal_ds.core.schema import UnifiedDocument, DataType, ProcessingStatus, Provenance

    docs = []
    for d in state.get("parsed_documents", []):
        ud = UnifiedDocument(
            data_type=DataType(d.get("data_type", "unknown")),
            status=ProcessingStatus(d.get("status", "done")),
            text_content=d.get("text_content", ""),
            schema_info=d.get("schema_info", {}),
            data_profile=d.get("data_profile", {}),
            provenance=Provenance(source_path=d.get("provenance", {}).get("source_path", ""))
        )
        docs.append(ud)

    for t in state.get("tabular_summaries", []):
        ud = UnifiedDocument(
            data_type=DataType.TABULAR,
            status=ProcessingStatus.DONE,
            text_content=t.get("sample", ""),
            schema_info={"columns": t.get("columns", []), "shape": t.get("shape", [])},
            data_profile=t.get("data_profile", {}),
            provenance=Provenance(source_path=t.get("source", ""))
        )
        docs.append(ud)

    session_id = state.get("session_id", str(uuid.uuid4())[:8])
    plan_result = run_planner(
        user_objective=state.get("user_query", ""),
        documents=docs,
        session_id=session_id,
    )

    tasks = plan_result.get("analysis_plan", [])
    return {
        "analysis_plan":  plan_result.get("final_plan", ""),
        "analysis_tasks": tasks,
        "hypotheses":     plan_result.get("hypotheses", []),
        "current_step":   0,
        "steps_total":    len(tasks),
    }


def _executor_node(state):
    from multimodal_ds.agents.code_execution_agent import CodeExecutionAgent
    from multimodal_ds.memory.agent_memory import AgentMemory
    from pathlib import Path

    tasks     = state.get("analysis_tasks", [])
    step_idx  = state.get("current_step", 0)

    if step_idx >= len(tasks):
        return state

    task       = tasks[step_idx]
    session_id = state.get("session_id", "default")

    retrieved = ""
    try:
        mem = AgentMemory(collection_name="doc_chunks")
        results = mem.retrieve(task.get("description", ""), n_results=4)
        retrieved = "\n\n".join(r["content"] for r in results)
    except Exception:
        pass

    data_files    = state.get("uploaded_files", [])
    tab_summaries = state.get("tabular_summaries", [])

    data_context_parts = []
    for fp in data_files:
        data_context_parts.append(f"Available file: {Path(fp).name}")
    for t in tab_summaries[:2]:
        cols = t.get("columns", [])[:20]
        data_context_parts.append(
            f"Table {Path(t['source']).name}: {t.get('shape')} rows×cols\n"
            f"Columns (first 20): {cols}"
        )
    if retrieved:
        data_context_parts.insert(0, f"Relevant document context:\n{retrieved}\n")

    agent = CodeExecutionAgent(session_id=session_id)
    exec_result = agent.execute(
        task_description=task.get("description", str(task)),
        data_context="\n".join(data_context_parts),
        file_paths=data_files,
    )

    new_output = f"Step {step_idx + 1} ({task.get('name', '?')}):\n{exec_result.get('output', '')}"
    new_error  = f"Step {step_idx + 1}: {exec_result['error'][:300]}" if exec_result.get("error") else None
    
    files = exec_result.get("files_created", [])
    new_vizs = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    new_arts = [f for f in files if f not in new_vizs]

    return {
        "current_step":    step_idx + 1,
        "code_outputs":    [new_output],
        "visualizations":  new_vizs,
        "saved_artifacts": new_arts,
        "errors":          [new_error] if new_error else [],
        "retrieved_context": retrieved,
    }


def _reviewer_node(state):
    """
    FIX: task_result dict now uses keys evaluation_agent.evaluate_task() actually reads:
      "name"           — task_result.get("name", "unknown_task")
      "success"        — task_result.get("success")
      "output_preview" — task_result.get("output_preview", "")
      "files_created"  — task_result.get("files_created", [])
    """
    from multimodal_ds.agents.evaluation_agent import EvaluationAgent

    tasks   = state.get("analysis_tasks", [])
    outputs = state.get("code_outputs", [])
    errors  = state.get("errors", [])

    task_results = []
    for i, (task, output) in enumerate(zip(tasks, outputs)):
        task_results.append({
            "name":           task.get("name", f"step_{i + 1}"),
            "success":        not any(f"Step {i + 1}:" in e for e in errors),
            "output_preview": output,
            "files_created":  [],
            "error":          "",
        })

    session_id = state.get("session_id", "default")
    eval_agent = EvaluationAgent(session_id=session_id)
    eval_report = eval_agent.evaluate_task_results(task_results)

    return {
        "eval_report": eval_report.to_dict() if hasattr(eval_report, "to_dict") else eval_report,
    }


def _retry_node(state):
    """
    Explicit node to increment retry count and log it.
    """
    count = state.get("retry_count", 0) + 1
    logger.warning(f"[Graph] Session retry triggered. New count: {count}")
    return {**state, "retry_count": count}


def _reporter_node(state):
    from multimodal_ds.agents.reporter import reporter_agent
    return reporter_agent(state)


# ── Conditional edges ────────────────────────────────────────────────────────

def _decide_ingestion_path(state) -> str:
    """
    FIX: Returns a single string key — not a list.
    Lists are only valid with Send() fan-out. Standard add_conditional_edges
    requires a single string matching one of the path_map keys.

    Priority: table > doc > image > audio > planner (no files)
    """
    flags    = state.get("_routing_flags", {})
    node_map = {
        "table": "tab_ingest",
        "doc":   "doc_ingest",
        "image": "img_ingest",
        "audio": "audio_ingest",
    }
    for kind, node in node_map.items():
        if flags.get(kind):
            return node
    return "planner"


def _decide_review_outcome(state) -> str:
    """
    Decide whether to:
    1. Continue to next task step (executor)
    2. Retry the whole session if overall failures (retry -> executor)
    3. Finish and report (reporter)
    """
    retry_count   = state.get("retry_count", 0)
    eval_report   = state.get("eval_report", {})
    overall_score = eval_report.get("overall_session_score", 10)
    has_failures  = eval_report.get("flagged_count", 0) > 0

    current = state.get("current_step", 0)
    total   = state.get("steps_total", 0)

    # 1. If we have more steps, keep going
    if current < total:
        return "executor"

    # 2. If we finished all steps but had critical failures, try a session-level retry
    if has_failures and retry_count < MAX_RETRIES and overall_score < 5:
        return "retry"

    # 3. Otherwise, we are done
    return "reporter"


# ── Graph builder ────────────────────────────────────────────────────────────

def build_graph(use_sqlite_checkpointer: bool = False, sqlite_path: str = "./checkpoints.db"):
    from langgraph.graph import StateGraph, END
    from multimodal_ds.core.state import AgentState

    builder = StateGraph(AgentState)

    builder.add_node("router",       _router_node)
    builder.add_node("doc_ingest",   _doc_ingest_node)
    builder.add_node("img_ingest",   _img_ingest_node)
    builder.add_node("audio_ingest", _audio_ingest_node)
    builder.add_node("tab_ingest",   _tab_ingest_node)
    builder.add_node("stats_val",    _stats_validation_node)
    builder.add_node("planner",      _planner_node)
    builder.add_node("executor",     _executor_node)
    builder.add_node("reviewer",     _reviewer_node)
    builder.add_node("retry",        _retry_node)
    builder.add_node("reporter",     _reporter_node)

    builder.set_entry_point("router")

    builder.add_conditional_edges(
        "router",
        _decide_ingestion_path,
        {
            "doc_ingest":   "doc_ingest",
            "img_ingest":   "img_ingest",
            "audio_ingest": "audio_ingest",
            "tab_ingest":   "tab_ingest",
            "planner":      "planner",
        }
    )

    for ingest_node in ["doc_ingest", "img_ingest", "audio_ingest"]:
        builder.add_edge(ingest_node, "planner")

    builder.add_edge("tab_ingest", "stats_val")
    builder.add_edge("stats_val",  "planner")
    builder.add_edge("planner",    "executor")
    builder.add_edge("executor",   "reviewer")

    builder.add_conditional_edges(
        "reviewer",
        _decide_review_outcome,
        {"executor": "executor", "retry": "retry", "reporter": "reporter"}
    )

    builder.add_edge("retry", "executor")

    builder.add_edge("reporter", END)

    if use_sqlite_checkpointer:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            memory = SqliteSaver.from_conn_string(sqlite_path)
        except ImportError:
            from langgraph.checkpoint.memory import MemorySaver
            memory = MemorySaver()
    else:
        from langgraph.checkpoint.memory import MemorySaver
        memory = MemorySaver()

    return builder.compile(checkpointer=memory)


def make_initial_state(
    user_query: str,
    uploaded_files: list[str],
    session_id: Optional[str] = None,
) -> dict:
    return {
        "user_query":         user_query,
        "uploaded_files":     uploaded_files,
        "_routing_flags":     {},
        "parsed_documents":   [],
        "image_embeddings":   [],
        "audio_transcripts":  [],
        "tabular_summaries":  [],
        "statistical_report": {},
        "analysis_plan":      "",
        "analysis_tasks":     [],
        "hypotheses":         [],
        "current_step":       0,
        "steps_total":        0,
        "code_outputs":       [],
        "visualizations":     [],
        "errors":             [],
        "retry_count":        0,
        "vector_store_id":    "",
        "retrieved_context":  "",
        "eval_report":        {},
        "final_report":       "",
        "session_id":         session_id or str(uuid.uuid4())[:8],
        "messages":           [],
    }
