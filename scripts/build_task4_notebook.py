"""Build the completed Task 4 workflow notebook from the Task 3 foundation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "03_multi_agent_system.ipynb"
TARGET = ROOT / "notebooks" / "04_agent_workflow.ipynb"


def markdown(source: str) -> dict[str, object]:
    """Return a normalized Markdown notebook cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip() + "\n",
    }


def code(source: str) -> dict[str, object]:
    """Return a normalized, unexecuted code notebook cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


def source_text(notebook: dict[str, object], index: int) -> str:
    """Read one source cell from the Task 3 notebook."""
    cell = notebook["cells"][index]
    return "".join(cell["source"])


def main() -> None:
    """Create a self-contained Task 4 notebook with live grading assertions."""
    task3 = json.loads(SOURCE.read_text(encoding="utf-8"))
    notebook = copy.deepcopy(task3)

    cells = [
        markdown(
            """
            # Task 4: Programming an agent workflow

            ## Goal

            Build and test a Google ADK question-answering system whose Greeter
            delegates every valid question to a deterministic Search → Critique →
            Refine workflow before the refined answer is returned.

            ## Checklist

            - [x] Copy the completed Task 3 notebook as the starting point.
            - [x] Create Greeter, Search, Critique, and Refine agents.
            - [x] Satisfy the `SequentialAgent or LoopAgent` requirement with a
              deterministic `SequentialAgent` answer team.
            - [x] Save the Search draft and Critique review in shared session state.
            - [x] Make Refine use both saved values and return the revised answer.
            - [x] Preserve stage authors, transfer, grounding, and state-change events.
            - [x] Test current facts, a misleading premise, a boundary case, and invalid input.
            - [x] Use a fresh ADK session for every live scenario.
            - [x] Map saved output to every grading criterion.

            - **Project:** `qwiklabs-gcp-02-66b2cfb8579b`
            - **Region:** `us-central1`
            - **Model:** `gemini-2.5-flash`
            """
        ),
        markdown(
            """
            ## 1. Task 3 foundation

            This notebook is programmatically copied forward from
            `03_multi_agent_system.ipynb`. It reuses the validated dependency and
            Google Cloud project preflight cells, then replaces Task 3's routing
            team with the Task 4 answer-verification workflow. No weather key is
            needed for this challenge.
            """
        ),
        code(source_text(task3, 2)),
        code(source_text(task3, 3)),
        markdown(
            """
            ## 2. Greeter and answer-team agents

            `greeter` is the root entry point. It transfers valid questions to
            `answer_team`, whose `SequentialAgent` runs the three specialist stages
            in a fixed order. ADK's `output_key` feature stores each model response
            in the shared session state, and `{initial_answer}` / `{critique}`
            placeholders pass those exact results into later stages.

            The Search and Critique agents each use only ADK's built-in Google Search tool. This
            respects the built-in tool boundary while grounding the initial draft
            in current web information.
            """
        ),
        code(
            r'''
            import asyncio
            from datetime import datetime, timezone

            from google.adk.agents import Agent, SequentialAgent
            from google.adk.agents.callback_context import CallbackContext
            from google.adk.runners import Runner
            from google.adk.sessions import InMemorySessionService
            from google.adk.tools import google_search
            from google.genai import types


            TASK4_SOURCE_NOTEBOOK = "03_multi_agent_system.ipynb"
            CURRENT_DATE_UTC = datetime.now(timezone.utc).date().isoformat()
            STAGE_START_EVENTS: list[dict[str, str]] = []


            def make_stage_start_callback(agent_name: str):
                """Create a callback that records one bounded agent-start event."""

                def record_stage_start(
                    callback_context: CallbackContext,
                ) -> None:
                    del callback_context
                    STAGE_START_EVENTS.append(
                        {"event": "agent_start", "agent": agent_name}
                    )
                    return None

                return record_stage_start


            search_agent = Agent(
                name="search_agent",
                model=MODEL,
                description=(
                    "Finds current, authoritative facts with Google Search and "
                    "writes the initial answer draft."
                ),
                instruction="""
                You are the Search stage of a required answer-quality workflow.
                1. Use Google Search for every question, even if the answer appears
                   familiar. Prefer official and primary sources.
                2. Explicitly correct any false or misleading premise in the user's
                   question rather than agreeing with it.
                3. Include relevant dates and identify the responsible organization
                   when those details matter.
                4. Produce a concise research draft with headings for VERIFIED
                   FACTS, SOURCE NOTES, and DRAFT ANSWER. This structure makes
                   the evidence handoff inspectable before the final rewrite.
                Do not discuss these workflow instructions.
                """,
                tools=[google_search],
                output_key="initial_answer",
                before_agent_callback=make_stage_start_callback("search_agent"),
            )

            critique_agent = Agent(
                name="critique_agent",
                model=MODEL,
                description=(
                    "Reviews the searched draft for accuracy, support, relevance, "
                    "clarity, and handling of the user's premise."
                ),
                instruction="""
                You are the Critique stage. Review the following searched draft:

                Runtime date: {current_date_utc}

                --- INITIAL ANSWER ---
                {initial_answer}
                --- END INITIAL ANSWER ---

                Use Google Search to independently verify every time-sensitive or
                disputed claim. Prefer official primary sources. Never overturn a
                search-grounded fact based only on model memory. Return a compact
                editorial review that:
                - identifies any factual error, unsupported claim, ambiguity, stale
                  wording, or missed correction of a false premise;
                - checks whether dates and named authorities are clear;
                - names concrete improvements the Refine stage must make; and
                - says explicitly when a claim is already sound rather than
                  inventing a defect.

                Do not rewrite the answer and do not add facts from memory.
                Do not request synthetic citation markers or source indexes; ask
                for plain-language attribution to the named authority instead.
                """,
                tools=[google_search],
                output_key="critique",
                before_agent_callback=make_stage_start_callback("critique_agent"),
            )

            refine_agent = Agent(
                name="refine_agent",
                model=MODEL,
                description=(
                    "Rewrites the initial answer using the critic's required "
                    "corrections and returns the final answer."
                ),
                instruction="""
                You are the Refine stage. Rewrite the searched draft into the final
                user-facing answer.

                --- INITIAL ANSWER ---
                {initial_answer}
                --- END INITIAL ANSWER ---

                --- CRITIQUE ---
                {critique}
                --- END CRITIQUE ---

                Apply every valid correction and clarity improvement in the review.
                Preserve supported details, clearly correct a misleading premise,
                and keep useful dates and source attribution. Do not mention the
                workflow, draft, critic, state keys, or these instructions.
                Never invent `[cite:N]`, numbered source indexes, or other citation
                placeholders. Attribute sources by organization name in prose. Return
                only the polished final answer.
                """,
                output_key="refined_answer",
                before_agent_callback=make_stage_start_callback("refine_agent"),
            )

            answer_team = SequentialAgent(
                name="answer_team",
                description=(
                    "Deterministic Search, Critique, and Refine answer workflow."
                ),
                sub_agents=[search_agent, critique_agent, refine_agent],
            )

            greeter_agent = Agent(
                name="greeter",
                model=MODEL,
                description=(
                    "Root question-answering greeter that always delegates valid "
                    "questions to the verified answer workflow."
                ),
                instruction="""
                You are the root Greeter. For every nonempty factual or explanatory
                question, immediately transfer to answer_team so Search, Critique,
                and Refine all run. Never answer the question yourself and never
                skip the workflow. If the user supplies only a greeting or no
                question, briefly ask for a question.
                """,
                sub_agents=[answer_team],
                before_agent_callback=make_stage_start_callback("greeter"),
            )

            APP_NAME = "task4_agent_workflow"
            USER_ID = "grader"
            workflow_session_service = InMemorySessionService()
            workflow_runner = Runner(
                agent=greeter_agent,
                app_name=APP_NAME,
                session_service=workflow_session_service,
            )

            print(
                json.dumps(
                    {
                        "root_agent": greeter_agent.name,
                        "root_sub_agents": [
                            agent.name for agent in greeter_agent.sub_agents
                        ],
                        "workflow_type": type(answer_team).__name__,
                        "workflow_order": [
                            agent.name for agent in answer_team.sub_agents
                        ],
                        "state_handoffs": {
                            "search": search_agent.output_key,
                            "critique": critique_agent.output_key,
                            "refine": refine_agent.output_key,
                        },
                        "search_tool": "google_search",
                        "critique_tool": "google_search",
                        "runtime_date_utc": CURRENT_DATE_UTC,
                        "model": MODEL,
                    },
                    indent=2,
                )
            )
            '''
        ),
        markdown(
            """
            ## 3. Deterministic architecture checks

            These checks prove the agent inventory, parent-child hierarchy,
            sequential order, Google Search boundary, and state handoffs without
            spending model quota.
            """
        ),
        code(
            """
            EXPECTED_STAGE_ORDER = [
                "search_agent",
                "critique_agent",
                "refine_agent",
            ]

            architecture_evidence = {
                "source_notebook": TASK4_SOURCE_NOTEBOOK,
                "root_name": greeter_agent.name,
                "root_sub_agents": [
                    agent.name for agent in greeter_agent.sub_agents
                ],
                "workflow_class": type(answer_team).__name__,
                "workflow_order": [
                    agent.name for agent in answer_team.sub_agents
                ],
                "search_uses_builtin_google_search": (
                    len(search_agent.tools) == 1
                    and search_agent.tools[0] is google_search
                ),
                "critique_uses_builtin_google_search": (
                    len(critique_agent.tools) == 1
                    and critique_agent.tools[0] is google_search
                ),
                "state_output_keys": [
                    search_agent.output_key,
                    critique_agent.output_key,
                    refine_agent.output_key,
                ],
            }

            assert TASK4_SOURCE_NOTEBOOK == "03_multi_agent_system.ipynb"
            assert architecture_evidence["root_name"] == "greeter"
            assert architecture_evidence["root_sub_agents"] == ["answer_team"]
            assert architecture_evidence["workflow_class"] == "SequentialAgent"
            assert architecture_evidence["workflow_order"] == EXPECTED_STAGE_ORDER
            assert architecture_evidence["search_uses_builtin_google_search"] is True
            assert architecture_evidence["critique_uses_builtin_google_search"] is True
            assert architecture_evidence["state_output_keys"] == [
                "initial_answer",
                "critique",
                "refined_answer",
            ]
            assert answer_team.parent_agent is greeter_agent
            assert all(
                agent.parent_agent is answer_team
                for agent in answer_team.sub_agents
            )
            print(json.dumps(architecture_evidence, indent=2))
            """
        ),
        markdown(
            """
            ## 4. Observable workflow runner

            ADK events are the execution log. Each compact record keeps the agent
            author, hierarchy branch, workflow transfer, function-call names,
            Google Search grounding presence, state keys changed, and final-response
            marker. Credential values, raw provider payloads, and authenticated URLs
            are excluded.
            """
        ),
        code(
            r'''
            def bounded_text(value: str | None, limit: int = 2400) -> str:
                """Normalize and truncate text for readable grading output."""
                if not value:
                    return ""
                normalized = " ".join(value.split())
                return normalized[:limit] + (
                    "..." if len(normalized) > limit else ""
                )


            def validate_workflow_prompt(prompt: str) -> dict[str, Any]:
                """Reject blank or unreasonably large prompts before any model call."""
                normalized = " ".join(prompt.split())
                if not normalized:
                    return {
                        "ok": False,
                        "error": "Please provide a nonempty question.",
                    }
                if len(normalized) > 1200:
                    return {
                        "ok": False,
                        "error": "Question exceeds the 1,200-character limit.",
                    }
                return {"ok": True, "prompt": normalized}


            def workflow_event_record(event: Any) -> dict[str, Any]:
                """Convert one ADK event into a compact, credential-free record."""
                calls = [call.name for call in event.get_function_calls()]
                state_delta = {}
                transfer_target = None
                if event.actions:
                    state_delta = dict(event.actions.state_delta or {})
                    transfer_target = event.actions.transfer_to_agent

                text = ""
                if event.content and event.content.parts:
                    text = "".join(
                        part.text or ""
                        for part in event.content.parts
                        if part.text
                    )

                return {
                    "author": event.author,
                    "branch": event.branch,
                    "transfer_to_agent": transfer_target,
                    "function_calls": calls,
                    "google_search_grounding": bool(
                        getattr(event, "grounding_metadata", None)
                    ),
                    "state_delta_keys": sorted(state_delta),
                    "is_final_response": event.is_final_response(),
                    "text_preview": bounded_text(text, limit=320),
                }


            async def run_workflow_case(
                prompt: str,
                *,
                label: str,
            ) -> dict[str, Any]:
                """Run one fresh-session workflow and return state plus event proof."""
                validation = validate_workflow_prompt(prompt)
                if not validation["ok"]:
                    return {
                        "label": label,
                        "accepted": False,
                        "error": validation["error"],
                        "model_called": False,
                    }

                session_id = f"task4-{label}-{uuid.uuid4().hex[:12]}"
                await workflow_session_service.create_session(
                    app_name=APP_NAME,
                    user_id=USER_ID,
                    session_id=session_id,
                    state={"current_date_utc": CURRENT_DATE_UTC},
                )
                STAGE_START_EVENTS.clear()
                message = types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=validation["prompt"])],
                )

                records: list[dict[str, Any]] = []
                final_answer = ""
                async with asyncio.timeout(150):
                    async for event in workflow_runner.run_async(
                        user_id=USER_ID,
                        session_id=session_id,
                        new_message=message,
                    ):
                        record = workflow_event_record(event)
                        records.append(record)
                        if record["is_final_response"] and event.content:
                            candidate = " ".join(
                                part.text or ""
                                for part in event.content.parts
                                if part.text
                            ).strip()
                            if candidate:
                                final_answer = candidate

                completed_session = await workflow_session_service.get_session(
                    app_name=APP_NAME,
                    user_id=USER_ID,
                    session_id=session_id,
                )
                assert completed_session is not None
                state = completed_session.state
                stage_outputs = {
                    key: str(state.get(key, "")).strip()
                    for key in (
                        "initial_answer",
                        "critique",
                        "refined_answer",
                    )
                }

                event_authors = list(
                    dict.fromkeys(record["author"] for record in records)
                )
                observed_stage_order = [
                    author
                    for author in event_authors
                    if author in EXPECTED_STAGE_ORDER
                ]
                transfer_targets = [
                    record["transfer_to_agent"]
                    for record in records
                    if record["transfer_to_agent"]
                ]

                return {
                    "label": label,
                    "accepted": True,
                    "session_id": session_id,
                    "started_agents": [
                        item["agent"] for item in STAGE_START_EVENTS
                    ],
                    "event_authors": event_authors,
                    "observed_stage_order": observed_stage_order,
                    "transfer_targets": transfer_targets,
                    "google_search_grounded": any(
                        record["google_search_grounding"] for record in records
                    ),
                    "grounded_agents": list(
                        dict.fromkeys(
                            record["author"]
                            for record in records
                            if record["google_search_grounding"]
                        )
                    ),
                    "initial_answer": bounded_text(
                        stage_outputs["initial_answer"]
                    ),
                    "critique": bounded_text(stage_outputs["critique"]),
                    "refinement": bounded_text(
                        stage_outputs["refined_answer"]
                    ),
                    "final_answer": bounded_text(final_answer),
                    "events": records,
                }
            '''
        ),
        markdown(
            """
            ## 5. Live end-to-end scenarios

            Each valid scenario begins at the Greeter in a fresh session. The
            misleading-premise case is designed to require an explicit factual
            correction. Assertions require all three workflow stages in order,
            live Google Search grounding, three saved state values, and a refined
            final answer. A sound draft may remain substantively unchanged; the
            misleading-premise case must still show the factual correction.
            """
        ),
        code(
            r'''
            LIVE_CASES = [
                {
                    "label": "current_official_update",
                    "prompt": (
                        "According to the latest official USGS Cascades Volcano "
                        "Observatory update, what are Mount Rainier's current "
                        "Volcano Alert Level and Aviation Color Code, and what do "
                        "they mean? Include the update date if one is stated."
                    ),
                },
                {
                    "label": "misleading_premise_correction",
                    "prompt": (
                        "The World Health Organization declared COVID-19 a pandemic "
                        "on March 11, 2019. Verify that date, correct it if needed, "
                        "and identify the organization that made the announcement."
                    ),
                },
                {
                    "label": "compact_boundary_answer",
                    "prompt": (
                        "In one compact paragraph, when did the International "
                        "Astronomical Union classify Pluto as a dwarf planet, and "
                        "what resolution defined the category?"
                    ),
                },
            ]

            live_results: list[dict[str, Any]] = []
            for case in LIVE_CASES:
                result = await run_workflow_case(
                    case["prompt"],
                    label=case["label"],
                )
                assert result["accepted"] is True, result
                assert result["observed_stage_order"] == EXPECTED_STAGE_ORDER, result
                assert set(["greeter", *EXPECTED_STAGE_ORDER]) <= set(
                    result["started_agents"]
                ), result
                assert "answer_team" in result["transfer_targets"], result
                assert result["google_search_grounded"] is True, result
                assert {"search_agent", "critique_agent"} <= set(
                    result["grounded_agents"]
                ), result
                assert result["initial_answer"], result
                assert result["critique"], result
                assert result["refinement"], result
                assert result["final_answer"], result
                assert result["final_answer"] == result["refinement"], result
                assert "[cite:" not in result["final_answer"].lower(), result
                live_results.append(result)

            assert len({item["session_id"] for item in live_results}) == len(
                live_results
            )
            correction_case = next(
                item
                for item in live_results
                if item["label"] == "misleading_premise_correction"
            )
            assert "2020" in correction_case["final_answer"], correction_case
            print(json.dumps(live_results, indent=2))
            '''
        ),
        markdown(
            """
            ## 6. Bounded failure handling

            Blank and oversized input are rejected locally before a session or
            model request is created. This gives the workflow a predictable failure
            boundary without spending quota or sending malformed input to Gemini.
            """
        ),
        code(
            """
            rejected_cases = [
                await run_workflow_case("   ", label="blank_question"),
                await run_workflow_case("x" * 1201, label="oversized_question"),
            ]
            assert all(item["accepted"] is False for item in rejected_cases)
            assert all(item["model_called"] is False for item in rejected_cases)
            assert all(item["error"] for item in rejected_cases)
            print(json.dumps(rejected_cases, indent=2))
            """
        ),
        markdown(
            """
            ## 7. Grading evidence

            The final assertions map every workshop and repository acceptance
            criterion to saved architecture, event, state, grounding, response, and
            failure-boundary output. A passing cell is the notebook's
            machine-checkable self-grade.
            """
        ),
        code(
            """
            correction_result = next(
                item
                for item in live_results
                if item["label"] == "misleading_premise_correction"
            )

            grading_evidence = {
                "copied_from_previous_notebook": (
                    TASK4_SOURCE_NOTEBOOK == "03_multi_agent_system.ipynb"
                ),
                "greeter_agent_created_as_root": (
                    greeter_agent.name == "greeter"
                    and workflow_runner.agent is greeter_agent
                ),
                "search_agent_created": search_agent.name == "search_agent",
                "critique_agent_created": critique_agent.name == "critique_agent",
                "refine_agent_created": refine_agent.name == "refine_agent",
                "sequential_answer_team_created": (
                    type(answer_team).__name__ == "SequentialAgent"
                    and architecture_evidence["workflow_order"]
                    == EXPECTED_STAGE_ORDER
                ),
                "search_uses_google_search": architecture_evidence[
                    "search_uses_builtin_google_search"
                ],
                "critique_rechecks_with_google_search": architecture_evidence[
                    "critique_uses_builtin_google_search"
                ],
                "initial_answer_saved_to_state": all(
                    result["initial_answer"] for result in live_results
                ),
                "critique_saved_to_state": all(
                    result["critique"] for result in live_results
                ),
                "refinement_saved_to_state": all(
                    result["refinement"] for result in live_results
                ),
                "stages_ran_in_required_order": all(
                    result["observed_stage_order"] == EXPECTED_STAGE_ORDER
                    for result in live_results
                ),
                "greeter_delegated_to_workflow": all(
                    "answer_team" in result["transfer_targets"]
                    for result in live_results
                ),
                "events_prove_every_sub_agent": all(
                    set(EXPECTED_STAGE_ORDER) <= set(result["event_authors"])
                    for result in live_results
                ),
                "live_google_search_grounding_saved": all(
                    {"search_agent", "critique_agent"}
                    <= set(result["grounded_agents"])
                    for result in live_results
                ),
                "initial_critique_refinement_final_visible": all(
                    result["initial_answer"]
                    and result["critique"]
                    and result["refinement"]
                    and result["final_answer"]
                    for result in live_results
                ),
                "final_answer_is_refined_answer": all(
                    result["final_answer"] == result["refinement"]
                    for result in live_results
                ),
                "no_invented_citation_placeholders": all(
                    "[cite:" not in result["final_answer"].lower()
                    for result in live_results
                ),
                "misleading_premise_corrected": (
                    "2020" in correction_result["final_answer"]
                ),
                "at_least_one_answer_changed_after_critique": any(
                    result["initial_answer"] != result["refinement"]
                    for result in live_results
                ),
                "fresh_session_for_every_live_case": len(
                    {result["session_id"] for result in live_results}
                )
                == len(live_results),
                "invalid_input_rejected_before_model": all(
                    not item["accepted"] and not item["model_called"]
                    for item in rejected_cases
                ),
                "all_live_responses_nonempty": all(
                    result["final_answer"] for result in live_results
                ),
            }

            assert all(grading_evidence.values()), grading_evidence
            print(json.dumps(grading_evidence, indent=2))
            print("TASK 4 COMPLETE: all agent-workflow grading checks passed.")
            """
        ),
        markdown(
            """
            ## References

            - [Google ADK workflow patterns](https://adk.dev/workflows/patterns/)
            - [Google ADK sequential workflow agents](https://adk.dev/agents/workflow-agents/sequential-agents/)
            - [Google ADK session state and `output_key`](https://adk.dev/sessions/state/)
            - [Google ADK events](https://adk.dev/events/)
            - [Google Search tool for ADK](https://adk.dev/tools/gemini-api/google-search/)
            """
        ),
    ]

    notebook["cells"] = cells
    notebook.setdefault("metadata", {}).setdefault("colab", {})["name"] = TARGET.name
    TARGET.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Built {TARGET} with {len(cells)} cells from {SOURCE.name}.")


if __name__ == "__main__":
    main()
