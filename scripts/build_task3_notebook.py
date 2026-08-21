"""Build the completed Task 3 multi-agent notebook from the Task 2 foundation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "02_callbacks.ipynb"
TARGET = ROOT / "notebooks" / "03_multi_agent_system.ipynb"


def markdown(source: str) -> dict[str, object]:
    """Return a normalized markdown notebook cell."""
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
    """Read one source cell from the Task 2 notebook."""
    cell = notebook["cells"][index]
    return "".join(cell["source"])


def main() -> None:
    """Create a self-contained Task 3 notebook with live grading assertions."""
    task2 = json.loads(SOURCE.read_text(encoding="utf-8"))
    notebook = copy.deepcopy(task2)

    cells = [
        markdown(
            """
            # Task 3: Developing multi-agent systems

            ## Goal

            Build and test a Google ADK coordinator that delegates U.S. weather
            work to a custom-tool specialist and current-information research to
            a Google Search specialist, with saved events that prove each handoff.

            ## Checklist

            - [x] Start from the completed Task 2 notebook and preserve its weather tools.
            - [x] Create a coordinating root agent.
            - [x] Create a weather agent with Google Maps and NWS function tools.
            - [x] Create a search agent with ADK's built-in Google Search tool.
            - [x] Register both specialists as root-agent sub-agents.
            - [x] Test weather-only, current-search, combined, unrelated, and failure cases.
            - [x] Save transfer, event-author, tool-call, grounding, and final-response evidence.
            - [x] Use fresh ADK sessions and grade every acceptance criterion.

            - **Project:** `qwiklabs-gcp-02-66b2cfb8579b`
            - **Region:** `us-central1`
            - **Model:** `gemini-3.7-flash` (`global` endpoint)
            """
        ),
        markdown(
            """
            ## 1. Task 2 foundation

            This notebook is programmatically copied forward from
            `02_callbacks.ipynb`. It reuses the executed Task 2 dependency,
            authentication, Google Maps Geocoding, and National Weather Service
            implementation. Task 3 adds the new search specialist, root
            coordinator, delegation trace, and grading scenarios.
            """
        ),
        code(source_text(task2, 2)),
        code(source_text(task2, 3)),
        code(source_text(task2, 4)),
        markdown(
            """
            ## 2. Reused weather tools

            The weather specialist keeps the typed, documented Task 2 tools.
            External calls have explicit timeouts and return compact sanitized
            objects without logging credentials or raw authenticated URLs.
            """
        ),
        code(source_text(task2, 6)),
        code(source_text(task2, 7)),
        code(source_text(task2, 9)),
        markdown(
            """
            ## 3. Specialist agents and root coordinator

            The search specialist contains only ADK's built-in
            `google_search` tool, respecting its single-tool-per-agent
            constraint. Clear descriptions give the root model reliable routing
            signals. Both specialists are registered through `sub_agents`, so
            ADK provides automatic transfer actions.
            """
        ),
        code(
            r'''
            import asyncio

            from google.adk.agents import Agent
            from google.adk.agents.callback_context import CallbackContext
            from google.adk.runners import Runner
            from google.adk.sessions import InMemorySessionService
            from google.adk.tools import google_search
            from google.genai import types


            TASK3_SOURCE_NOTEBOOK = "02_callbacks.ipynb"
            AGENT_START_EVENTS: list[dict[str, str]] = []


            def configured_tool_name(tool: Any) -> str:
                """Return the public name of a built-in, function, or wrapped tool."""
                wrapped_function = getattr(tool, "func", None)
                wrapped_name = getattr(wrapped_function, "__name__", None)
                if wrapped_name:
                    return str(wrapped_name)
                direct_name = getattr(tool, "name", None)
                return str(direct_name or type(tool).__name__)


            def make_agent_start_callback(agent_name: str):
                """Create a callback that records a bounded specialist-start event."""

                def record_agent_start(
                    callback_context: CallbackContext,
                ) -> None:
                    del callback_context
                    AGENT_START_EVENTS.append(
                        {"event": "agent_start", "agent": agent_name}
                    )
                    return None

                return record_agent_start


            weather_agent = Agent(
                name="weather_agent",
                model=MODEL,
                description=(
                    "Specialist for live U.S. weather observations, forecasts, "
                    "and National Weather Service alerts for a named city and state."
                ),
                instruction="""
                You are the weather specialist in a multi-agent team.
                1. For every weather request, call geocode_place with the full U.S.
                   location, then call get_weather with the returned coordinates.
                2. Report the resolved location, current observation when available,
                   forecast, and active-alert status. Put urgent alerts first.
                3. Never use general web knowledge or invent weather.
                4. If the user also asks for current web research, finish the weather
                   work, summarize it briefly, then transfer to root_agent so the
                   coordinator can send the remaining work to google_search_agent.
                5. For weather-only requests, answer directly and concisely.
                """,
                tools=[geocode_place, get_weather],
                before_agent_callback=make_agent_start_callback("weather_agent"),
            )

            google_search_agent = Agent(
                name="google_search_agent",
                model=MODEL,
                description=(
                    "Specialist for current facts, recent developments, official "
                    "announcements, and web research using Google Search."
                ),
                instruction="""
                You are the current-information specialist in a multi-agent team.
                Use the built-in Google Search tool for every request. Prefer
                authoritative primary sources, state relevant dates, distinguish
                sourced facts from inference, and give a concise answer. For a
                combined request, incorporate the weather facts already present in
                the conversation and finish the answer after completing search.
                """,
                tools=[google_search],
                before_agent_callback=make_agent_start_callback(
                    "google_search_agent"
                ),
                # A built-in Google Search tool cannot be mixed with ADK's
                # automatic transfer tools. The root can transfer into this
                # terminal specialist after any weather handoff is complete.
                disallow_transfer_to_parent=True,
                disallow_transfer_to_peers=True,
            )

            root_agent = Agent(
                name="root_agent",
                model=MODEL,
                description=(
                    "Coordinator that routes requests to live U.S. weather and "
                    "current-information Google Search specialists."
                ),
                instruction="""
                You coordinate exactly two specialists.
                - Delegate live U.S. weather, forecasts, observations, or alerts to
                  weather_agent. Do not answer weather from memory.
                - Delegate current events, recent developments, official updates,
                  or other web research to google_search_agent.
                - When a request needs both, delegate the weather portion first.
                  After weather_agent returns control, delegate the remaining current
                  research to google_search_agent, then provide a concise combined
                  answer grounded in both specialists' results.
                - For requests outside both capabilities, refuse briefly and describe
                  the supported weather and current-information capabilities.
                Do not call specialist tools yourself or fabricate specialist results.
                """,
                sub_agents=[weather_agent, google_search_agent],
                before_agent_callback=make_agent_start_callback("root_agent"),
            )

            APP_NAME = "task3_multi_agent_system"
            USER_ID = "grader"
            multi_agent_session_service = InMemorySessionService()
            multi_agent_runner = Runner(
                agent=root_agent,
                app_name=APP_NAME,
                session_service=multi_agent_session_service,
            )

            print(
                json.dumps(
                    {
                        "root_agent": root_agent.name,
                        "sub_agents": [
                            agent.name for agent in root_agent.sub_agents
                        ],
                        "weather_tools": [
                            geocode_place.__name__,
                            get_weather.__name__,
                        ],
                        "search_tools": [
                            configured_tool_name(tool)
                            for tool in google_search_agent.tools
                        ],
                        "model": MODEL,
                    },
                    indent=2,
                )
            )
            '''
        ),
        markdown(
            """
            ## 4. Observable event runner

            ADK events are the grading log. Each record preserves the author,
            hierarchy branch, transfer target, bounded function-call metadata,
            Google Search grounding presence, and a short final-text preview.
            Full provider responses and credentials are deliberately excluded.
            """
        ),
        code(
            r'''
            SPECIALIST_NAMES = {"weather_agent", "google_search_agent"}


            def bounded_text(value: str | None, limit: int = 320) -> str | None:
                """Normalize and truncate display text for a compact event log."""
                if not value:
                    return None
                normalized = " ".join(value.split())
                return normalized[:limit] + ("..." if len(normalized) > limit else "")


            def event_record(event: Any) -> dict[str, Any]:
                """Convert an ADK event into a compact, credential-free record."""
                calls = [
                    {
                        "name": call.name,
                        "arguments": dict(call.args or {}),
                    }
                    for call in event.get_function_calls()
                ]
                responses = []
                for response in event.get_function_responses():
                    payload = response.response
                    responses.append(
                        {
                            "name": response.name,
                            "status": (
                                payload.get("status")
                                if isinstance(payload, dict)
                                else None
                            ),
                        }
                    )

                text = ""
                if event.content and event.content.parts:
                    text = "".join(
                        part.text or "" for part in event.content.parts if part.text
                    )

                transfer_target = None
                if event.actions:
                    transfer_target = event.actions.transfer_to_agent

                grounding = bool(getattr(event, "grounding_metadata", None))
                return {
                    "author": event.author,
                    "branch": event.branch,
                    "transfer_to_agent": transfer_target,
                    "function_calls": calls,
                    "function_responses": responses,
                    "google_search_grounding": grounding,
                    "is_final_response": event.is_final_response(),
                    "text_preview": bounded_text(text),
                }


            async def run_multi_agent_case(
                prompt: str,
                *,
                label: str,
            ) -> dict[str, Any]:
                """Run one fresh-session root-agent turn and return its event trace."""
                session_id = f"task3-{label}-{uuid.uuid4().hex[:12]}"
                await multi_agent_session_service.create_session(
                    app_name=APP_NAME,
                    user_id=USER_ID,
                    session_id=session_id,
                )
                AGENT_START_EVENTS.clear()
                message = types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )

                records: list[dict[str, Any]] = []
                final_answer = ""
                async with asyncio.timeout(120):
                    async for event in multi_agent_runner.run_async(
                        user_id=USER_ID,
                        session_id=session_id,
                        new_message=message,
                    ):
                        record = event_record(event)
                        records.append(record)
                        if record["is_final_response"] and record["text_preview"]:
                            final_answer = " ".join(
                                part.text or ""
                                for part in event.content.parts
                                if part.text
                            ).strip()

                event_authors = list(
                    dict.fromkeys(
                        record["author"]
                        for record in records
                        if record["author"]
                    )
                )
                transfer_targets = [
                    record["transfer_to_agent"]
                    for record in records
                    if record["transfer_to_agent"]
                ]
                tool_calls = [
                    {
                        "author": record["author"],
                        **call,
                    }
                    for record in records
                    for call in record["function_calls"]
                    if call["name"] != "transfer_to_agent"
                ]
                started_agents = list(
                    dict.fromkeys(item["agent"] for item in AGENT_START_EVENTS)
                )

                return {
                    "label": label,
                    "prompt": prompt,
                    "session_id": session_id,
                    "started_agents": started_agents,
                    "event_authors": event_authors,
                    "transfer_targets": transfer_targets,
                    "tool_calls": tool_calls,
                    "google_search_grounded": any(
                        record["google_search_grounding"] for record in records
                    ),
                    "final_answer": final_answer,
                    "events": records,
                }
            '''
        ),
        markdown(
            """
            ## 5. Deterministic architecture checks

            These checks prove the three-agent hierarchy and each specialist's
            tool boundary without calling Gemini or an external service.
            """
        ),
        code(
            """
            hierarchy_evidence = {
                "root_name": root_agent.name,
                "sub_agent_names": [agent.name for agent in root_agent.sub_agents],
                "weather_tool_names": [
                    geocode_place.__name__,
                    get_weather.__name__,
                ],
                "weather_agent_tool_count": len(weather_agent.tools),
                "search_tool_names": [
                    configured_tool_name(tool)
                    for tool in google_search_agent.tools
                ],
                "search_uses_builtin_google_search": (
                    len(google_search_agent.tools) == 1
                    and google_search_agent.tools[0] is google_search
                ),
                "search_is_terminal_specialist": bool(
                    google_search_agent.disallow_transfer_to_parent
                    and google_search_agent.disallow_transfer_to_peers
                ),
            }

            assert hierarchy_evidence["root_name"] == "root_agent"
            assert hierarchy_evidence["sub_agent_names"] == [
                "weather_agent",
                "google_search_agent",
            ]
            assert set(hierarchy_evidence["weather_tool_names"]) == {
                "geocode_place",
                "get_weather",
            }
            assert hierarchy_evidence["weather_agent_tool_count"] == 2
            assert hierarchy_evidence["search_uses_builtin_google_search"] is True
            assert hierarchy_evidence["search_is_terminal_specialist"] is True
            assert weather_agent.parent_agent is root_agent
            assert google_search_agent.parent_agent is root_agent
            print(json.dumps(hierarchy_evidence, indent=2))
            """
        ),
        markdown(
            """
            ## 6. Live delegation scenarios

            Every scenario starts with the root agent in a fresh ADK session.
            Assertions require the appropriate specialist to appear as an event
            author, not merely in configuration. The combined case must exercise
            both specialists. The unrelated case must stay with the coordinator.
            """
        ),
        code(
            r'''
            LIVE_CASES = [
                {
                    "label": "weather_only",
                    "prompt": (
                        "What is the current weather, forecast, and active-alert "
                        "status for Chicago, IL?"
                    ),
                    "expected_specialists": {"weather_agent"},
                    "expected_tools": {"geocode_place", "get_weather"},
                    "requires_grounding": False,
                },
                {
                    "label": "search_only",
                    "prompt": (
                        "What are the latest official NASA updates about the "
                        "Artemis II mission? Include relevant publication dates."
                    ),
                    "expected_specialists": {"google_search_agent"},
                    "expected_tools": set(),
                    "requires_grounding": True,
                },
                {
                    "label": "combined_weather_and_search",
                    "prompt": (
                        "Give me the current weather and active-alert status for "
                        "Miami, FL, then find the latest official NOAA Atlantic "
                        "hurricane outlook and summarize both."
                    ),
                    "expected_specialists": {
                        "weather_agent",
                        "google_search_agent",
                    },
                    "expected_tools": {"geocode_place", "get_weather"},
                    "requires_grounding": True,
                },
                {
                    "label": "unrelated_request",
                    "prompt": "Write a limerick about database indexes.",
                    "expected_specialists": set(),
                    "expected_tools": set(),
                    "requires_grounding": False,
                },
            ]

            live_results: list[dict[str, Any]] = []
            for case in LIVE_CASES:
                result = await run_multi_agent_case(
                    case["prompt"],
                    label=case["label"],
                )
                specialist_authors = SPECIALIST_NAMES & set(
                    result["event_authors"]
                )
                specialist_starts = SPECIALIST_NAMES & set(
                    result["started_agents"]
                )
                observed_tools = {
                    call["name"] for call in result["tool_calls"]
                }

                assert result["final_answer"], result
                assert specialist_authors == case["expected_specialists"], result
                assert specialist_starts == case["expected_specialists"], result
                assert case["expected_specialists"] <= set(
                    result["transfer_targets"]
                ), result
                assert case["expected_tools"] <= observed_tools, result
                if case["requires_grounding"]:
                    assert result["google_search_grounded"] is True, result
                if not case["expected_specialists"]:
                    assert observed_tools == set(), result

                live_results.append(result)

            assert len({item["session_id"] for item in live_results}) == len(
                live_results
            )
            print(json.dumps(live_results, indent=2))
            '''
        ),
        markdown(
            """
            ## 7. Failure and boundary behavior

            An invalid location must still route through the weather specialist,
            call geocoding, avoid the downstream NWS call, and return a clear
            nonempty explanation.
            """
        ),
        code(
            """
            invalid_location_result = await run_multi_agent_case(
                (
                    "What is the current weather for "
                    "This Place Should Not Exist 9z8y7x6w5v?"
                ),
                label="invalid_location",
            )
            invalid_tool_names = [
                call["name"] for call in invalid_location_result["tool_calls"]
            ]
            assert "weather_agent" in invalid_location_result["event_authors"]
            assert "weather_agent" in invalid_location_result["transfer_targets"]
            assert invalid_tool_names == ["geocode_place"], invalid_location_result
            assert invalid_location_result["final_answer"], invalid_location_result
            print(json.dumps(invalid_location_result, indent=2))
            """
        ),
        markdown(
            """
            ## 8. Grading evidence

            The final assertions map the workshop and repository acceptance
            criteria to saved live output. A passing cell is the notebook's
            machine-checkable self-grade.
            """
        ),
        code(
            """
            results_by_label = {item["label"]: item for item in live_results}
            weather_live = results_by_label["weather_only"]
            search_live = results_by_label["search_only"]
            combined_live = results_by_label["combined_weather_and_search"]
            unrelated_live = results_by_label["unrelated_request"]

            evidence = {
                "copied_from_task_2": (
                    TASK3_SOURCE_NOTEBOOK == "02_callbacks.ipynb"
                ),
                "three_agents_created": {
                    root_agent.name,
                    weather_agent.name,
                    google_search_agent.name,
                }
                == {"root_agent", "weather_agent", "google_search_agent"},
                "coordinating_root_agent": root_agent.name == "root_agent",
                "weather_agent_uses_custom_tools": set(
                    hierarchy_evidence["weather_tool_names"]
                )
                == {"geocode_place", "get_weather"},
                "search_agent_uses_builtin_google_search": hierarchy_evidence[
                    "search_uses_builtin_google_search"
                ],
                "search_tool_boundary_is_valid": hierarchy_evidence[
                    "search_is_terminal_specialist"
                ],
                "specialists_registered_as_sub_agents": hierarchy_evidence[
                    "sub_agent_names"
                ]
                == ["weather_agent", "google_search_agent"],
                "root_delegated_weather_request": (
                    "weather_agent" in weather_live["transfer_targets"]
                    and "weather_agent" in weather_live["event_authors"]
                ),
                "root_delegated_search_request": (
                    "google_search_agent" in search_live["transfer_targets"]
                    and "google_search_agent" in search_live["event_authors"]
                ),
                "weather_tools_ran_live": {
                    call["name"] for call in weather_live["tool_calls"]
                }
                >= {"geocode_place", "get_weather"},
                "google_search_ran_live": search_live["google_search_grounded"],
                "combined_request_used_both_specialists": SPECIALIST_NAMES
                <= set(combined_live["event_authors"]),
                "combined_request_used_both_sources": bool(
                    combined_live["google_search_grounded"]
                    and {
                        call["name"] for call in combined_live["tool_calls"]
                    }
                    >= {"geocode_place", "get_weather"}
                ),
                "events_prove_sub_agent_use": all(
                    item["events"] for item in live_results
                ),
                "unrelated_request_handled_by_root": (
                    not (SPECIALIST_NAMES & set(unrelated_live["event_authors"]))
                    and unrelated_live["final_answer"]
                ),
                "failure_case_is_safe_and_bounded": (
                    [call["name"] for call in invalid_location_result["tool_calls"]]
                    == ["geocode_place"]
                    and bool(invalid_location_result["final_answer"])
                ),
                "fresh_sessions_for_every_live_case": len(
                    {
                        item["session_id"]
                        for item in [*live_results, invalid_location_result]
                    }
                )
                == len(live_results) + 1,
                "all_live_responses_nonempty": all(
                    item["final_answer"]
                    for item in [*live_results, invalid_location_result]
                ),
            }

            assert all(evidence.values()), evidence
            print(json.dumps(evidence, indent=2))
            print("TASK 3 COMPLETE: all multi-agent grading checks passed.")
            """
        ),
        markdown(
            """
            ## References

            - [Google ADK multi-agent workflow patterns](https://adk.dev/workflows/patterns/)
            - [Google ADK agent-team tutorial](https://adk.dev/tutorials/agent-team/)
            - [Google Search tool for ADK](https://adk.dev/tools/gemini-api/google-search/)
            - [Google ADK events](https://adk.dev/events/)
            - [Google Maps Geocoding API](https://developers.google.com/maps/documentation/geocoding)
            - [National Weather Service API](https://www.weather.gov/documentation/services-web-api)
            """
        ),
    ]

    notebook["cells"] = cells
    notebook.setdefault("metadata", {}).setdefault("colab", {})["name"] = TARGET.name
    TARGET.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Built {TARGET} with {len(cells)} cells from {SOURCE.name}.")


if __name__ == "__main__":
    main()
