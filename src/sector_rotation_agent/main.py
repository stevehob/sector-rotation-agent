import asyncio
import argparse
import logging
from datetime import date, datetime
from functools import partial
from pathlib import Path

import sector_rotation_agent.constants as const
from sector_rotation_agent.config import settings
from sector_rotation_agent.model_client import make_model_client
from sector_rotation_agent.trace import TraceLogger, configure_component_logging
from sector_rotation_agent.fred_query import FredMCPClient
from sector_rotation_agent.yfin_query import YFinMCPClient
from sector_rotation_agent.generate_hypotheses import generate_hypotheses
from sector_rotation_agent.historical_analogs import find_historical_analogs
from sector_rotation_agent.score_branch import score_branch
from sector_rotation_agent.synthesize import compute_sector_score, generate_report, report_to_pdf
from sector_rotation_agent.equity_agent import EquityAgent
from sector_rotation_agent.macro_agent import MacroAgent
from sector_rotation_agent.coordinator import Coordinator, llm_decompose_query
from sector_rotation_agent.audit import ResultAuditor
from sector_rotation_agent import fed_narrative_rag

async def amain(query: str, as_of: str):
    # Per-component log files (main / coordinator / model_client / trace) plus a
    # structured, per-run trace (the TraceLogger) capturing phase latencies and full
    # LLM-call detail. The trace is injected into the model client and the coordinator.
    configure_component_logging()
    trace = TraceLogger(run_id=f"{as_of or 'run'}-{datetime.now():%Y%m%dT%H%M%S}")
    log = logging.getLogger("sector_rotation_agent.main")
    trace.event("main", "run_start", query=query, as_of=as_of)
    log.info("Run %s started: as_of=%s, query=%r", trace.run_id, as_of, query)

    # Check and make sure our history data store is available
    stats_path = Path(const.NORM_STATS_PATH)
    db_path = Path(const.STORE_PATH)
    if not stats_path.is_file() or not db_path.is_dir() or not any(db_path.iterdir()):
        raise RuntimeError ("History data not found: Run python -m sector_rotation_agent.historical_analogs to seed first")
    
    #  ---------------  async block - run everything in here to make sure sessions remain open
    # open MCP client sessions
    mcp_servers = Path(const.MCP_SERVERS_PATH)
    async with FredMCPClient(mcp_servers / const.MCP_CLIENT_FRED) as fred, \
                YFinMCPClient(mcp_servers / const.MCP_CLIENT_YFIN) as yfin:
        
        # ---  Version 1  ----------------------------------------------------------  
        # One traced model client, shared by every LLM seam in the run -- the query
        # decomposer, the ToT hypothesis fan-out, the audit critic, and the report's
        # executive summary -- so the TraceLogger captures EVERY LLM call (prompts,
        # response, token usage, latency) in one place. Point this at a cloud client
        # (make_model_client("anthropic", trace=trace)) for the spec's hybrid split.
        #model_complete = make_model_client(trace=trace).complete

        # ---  Version 1.1  --------------------------------------------------------
        # Create multiple models from different sources (cloud/local)
        # This enables high-value actions like generate_hypothesis and 
        # generate_executive_summary to use 'expensive' models.
        # Lowever-value, small activities like run_critic_check and 
        # llm_decompose_query can use smaller, local models.
        # Alternatively, could also mix model types to give critic a distinct
        # perspective.
        # Based on ModelLocations: LOCAL_ONLY | MIXED | CLOUD_ONLY, construct the right models
        # summary_label records which model actually writes the executive summary, for the
        # report's run-metadata footer. The summary seam (write_report) uses cloud_model in
        # every mode, so the label must track cloud_model -- in LOCAL_ONLY that is the LOCAL
        # model, so a hardcoded cloud label would misreport which model wrote the brief and
        # corrupt model-comparison runs.
        match settings.model_location:
            case const.ModelLocations.LOCAL_ONLY:
                local_model = make_model_client(settings.local_model_service, model=settings.local_model, trace=trace).complete
                cloud_model = local_model
                summary_label = f"{settings.local_model_service}/{settings.local_model}"
            case const.ModelLocations.MIXED:
                cloud_model = make_model_client(settings.cloud_model_service, model=settings.cloud_model, trace=trace).complete
                local_model = make_model_client(settings.local_model_service, model=settings.local_model, trace=trace).complete
                summary_label = f"{settings.cloud_model_service}/{settings.cloud_model}"
            case const.ModelLocations.CLOUD_ONLY:
                cloud_model = make_model_client(settings.cloud_model_service, model=settings.cloud_model, trace=trace).complete
                local_model = cloud_model
                summary_label = f"{settings.cloud_model_service}/{settings.cloud_model}"
            case _:
                raise RuntimeError(f"Invalid model location setting: {settings.model_location}")
        
        


        # construct the agents. generate_hypotheses (the ToT fan-out) is bound to the
        # traced client so its LLM call lands in the per-run trace JSONL too -- not just
        # model_client.log -- the same partial-binding as the decomposer/critic/summary.
        macro_agent = MacroAgent(
            data_client=fred,
            generate_hypotheses=partial(generate_hypotheses, call_model=cloud_model),  # v1: call_model=model_complete
            find_historical_analogs=find_historical_analogs,
            score_branch=score_branch,
            find_fed_narrative=fed_narrative_rag.find_fed_narrative,
        )
        
        equity_agent = EquityAgent(
            data_client=yfin,
        )

        # Phase 1 audit layer: the pure-Python statistical checker plus a LIVE LLM critic
        # that cross-checks the regime call against the strongest analog / Fed passage.
        # The critic now runs through the traced client above (previously an untraced
        # default), so its calls show up in the trace too. check_corpus_freshness wires
        # guardrail #2 at the corpus level (a stale Fed corpus -> carried-forward caveat).
        audit = ResultAuditor(
            call_model=local_model,  # v1: call_model=model_complete
            check_freshness=fed_narrative_rag.check_corpus_freshness,
        )

        # The brief's opening summary is an LLM call during report assembly -- AFTER the
        # ReAct loop and reconciliation -- so it never goes through record_tool_call and
        # does NOT count against the 20-tool-call cap (guardrail #3). model_label/run_id
        # are bound here too (per-run constants, like call_model) so the report's footer
        # identifies which model wrote the summary and pairs the brief with its
        # trace-<run_id>.jsonl -- handy when comparing briefs across models.
        write_report = partial(
            generate_report,
            call_model=cloud_model,  # v1: call_model=model_complete
            model_label=summary_label,
            run_id=trace.run_id,
        )

        # LLM query decomposition (coordinator seam): extracts the horizon from the
        # question, with a deterministic regex fallback baked into llm_decompose_query.
        # It runs once before the loop and is not a recorded tool call (like the summary).
        decompose = partial(llm_decompose_query, call_model=cloud_model) # v1: call_model=model_complete

        # build the agent coordinator
        coordinator = Coordinator(
            macro_agent=macro_agent,
            equity_agent=equity_agent,
            score_sectors=compute_sector_score,
            audit=audit,
            generate_report=write_report,
            decompose=decompose,
            max_tool_calls=20,
            max_revision_cycles=3,
            trace=trace,
        )

        # run and get back results
        result = await coordinator.run(query, as_of=as_of, tickers=const.SECTOR_ETFS_LIST, period="5y")

    # Close out the trace: a one-line run summary (events, LLM calls, tokens, latency) to
    # the trace log, plus a final event and the per-run JSONL file path.
    trace.event("main", "run_complete", regime=result.regime.value,
                low_confidence=result.low_confidence)
    trace.summary()
    return result


def _prompt_pdf_export(report: str | None) -> None:
    """Offer to render the brief to PDF. Skips silently when there's no report or no
    interactive stdin, so piped / non-interactive runs aren't blocked."""
    if not report:
        return
    try:
        answer = input("\nWould you like a PDF version of this brief? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return  # non-interactive or aborted -> just skip the export
    if answer not in ("y", "yes"):
        return
    out_path = Path(const.STORE_PATH).parents[1] / "reports" / "sector_rotation_brief.pdf"
    try:
        written = report_to_pdf(report, out_path)
    except RuntimeError as err:  # markdown-pdf not installed
        print(f"Could not export PDF: {err}")
        return
    print(f"PDF written to: {written.resolve()}")

def _prompt_user_question(prompt: str = "") -> str:

    if prompt:
        user_input = input(prompt)
    else:
        print("Macro-Driven Sector Research Agent\n"
                "I analyze macroeconomic and equity-sector data to produce an intelligent, "
                "supportable, and audited response to your financial research questions.\n"
                "An example I'm prepared to analyze is something like: "
                "Which sectors should I overweight/underweight over the next 6 months?\n\n")

        #user_input = input(f"Please enter a question... ")
        # for testing purposes
        user_input = "Which sectors should I overweight/underweight over the next 6 months?"

    #TODO: input validation before return
    return user_input

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Macro-Driven Sector Research Agent")
    parser.add_argument("--q", type=str, nargs="?", default="", 
                            help="Macro-economic forecast question")
    parser.add_argument("--ao", type=str, nargs="?", default="",
                            help="Date in the format: YYYY-MM-DD")
    #parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")

    args = parser.parse_args()

    as_of = args.ao
    if not as_of:
        as_of = date.today().isoformat() # default to now
    user_question = args.q
    
    print(f"question: {user_question}")
    print(f"as-of: {as_of}")

    if user_question:
        print(f"Working to answer the following question: {str(user_question)}")
    else:
        user_question = _prompt_user_question("")

    print(f"Analyzing with data as of: {as_of}")
    print("Thinking ...")

    if user_question:
        result = asyncio.run(amain(user_question, as_of))
        print(result.report)
        _prompt_pdf_export(result.report)
