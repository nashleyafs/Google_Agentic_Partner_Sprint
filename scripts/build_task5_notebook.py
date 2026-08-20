"""Build the completed Task 5 Agent Engine deployment notebook."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "04_agent_workflow.ipynb"
TARGET = ROOT / "notebooks" / "05_deploy_agent.ipynb"


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


def main() -> None:
    """Create a self-contained Task 5 notebook with live deployment checks."""
    task4 = json.loads(SOURCE.read_text(encoding="utf-8"))
    notebook = copy.deepcopy(task4)

    cells = [
        markdown(
            """
            # Task 5: Deploying an ADK agent

            ## Goal

            Carry the Task 4 Search → Critique → Refine workflow into a
            deployment-safe Google ADK application, deploy it to Vertex AI Agent
            Engine, and add a bounded Creativity → Refine loop that improves the
            writing while keeping Refine as the final author.

            ## Checklist

            - [x] Copy the completed Task 4 notebook as the starting point.
            - [x] Recreate its Greeter, Search, Critique, and Refine workflow.
            - [x] Add a Creativity → Refine loop capped at three cycles.
            - [x] Keep Refine as the final author and factuality gate.
            - [x] Test the deployable application locally with a fresh ADK session.
            - [x] Initialize the current Vertex AI Agent Engine client.
            - [x] Deploy with explicit, pinned runtime requirements.
            - [x] Save the deployed Agent Engine resource name.
            - [x] Run and save a successful remote query in a fresh session.
            - [x] Reject invalid boundary input before any model request.
            - [x] Include guarded Cleanup code without deleting the graded resource.
            - [x] Map saved output to every grading criterion.

            - **Project:** `qwiklabs-gcp-02-66b2cfb8579b`
            - **Region:** `us-central1`
            - **Model:** `gemini-2.5-flash`
            """
        ),
        markdown(
            """
            ## 1. Task 4 foundation and dependencies

            This notebook is programmatically copied forward from
            `04_agent_workflow.ipynb`. The deployed agent keeps Task 4's verified
            answer architecture, extended with a bounded Creativity → Refine loop.
            Notebook-only callback lists and the local
            `Runner` are omitted from the deployment object so its serialized
            package contains only the ADK agents and built-in Google Search tool.

            The runtime versions below match the live Workbench environment and
            are pinned again in `DEPLOYMENT_REQUIREMENTS` for a reproducible Agent
            Engine build.
            """
        ),
        code(
            """
            import importlib.metadata
            import importlib.util
            import subprocess
            import sys


            EXPECTED_VERSIONS = {
                "google-cloud-aiplatform": "1.164.0",
                "google-adk": "1.39.0",
            }
            installed_versions = {
                package: importlib.metadata.version(package)
                for package in EXPECTED_VERSIONS
            }

            if installed_versions != EXPECTED_VERSIONS:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--quiet",
                        "google-cloud-aiplatform[agent_engines,adk]==1.164.0",
                        "google-adk==1.39.0",
                    ],
                    check=True,
                )
                installed_versions = {
                    package: importlib.metadata.version(package)
                    for package in EXPECTED_VERSIONS
                }

            assert installed_versions == EXPECTED_VERSIONS
            print(installed_versions)
            """
        ),
        markdown(
            """
            ## 2. Authentication and project preflight

            Application Default Credentials are used; no key or token is printed.
            A project mismatch is a hard stop before a model call, bucket creation,
            or deployment.
            """
        ),
        code(
            r'''
            from __future__ import annotations

            import json
            import os
            import subprocess
            import uuid
            from datetime import datetime, timezone
            from typing import Any

            import google.auth
            import vertexai
            from google.adk.agents import Agent, LoopAgent, SequentialAgent
            from google.adk.tools import google_search
            from vertexai import agent_engines


            EXPECTED_PROJECT = "qwiklabs-gcp-02-66b2cfb8579b"
            LOCATION = "us-central1"
            MODEL = "gemini-2.5-flash"
            CURRENT_DATE_UTC = datetime.now(timezone.utc).date().isoformat()


            def run_gcloud(
                arguments: list[str], timeout: int = 60
            ) -> subprocess.CompletedProcess[str]:
                """Run a bounded gcloud command without printing credentials."""
                return subprocess.run(
                    ["gcloud", *arguments],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )


            project_result = run_gcloud(["config", "get-value", "project"])
            detected_project = project_result.stdout.strip()
            _, adc_project = google.auth.default()
            observed_projects = {
                value for value in (detected_project, adc_project) if value
            }

            preflight = {
                "expected_project": EXPECTED_PROJECT,
                "gcloud_project": detected_project,
                "adc_project": adc_project,
                "location": LOCATION,
                "model": MODEL,
                "google_cloud_aiplatform_version": importlib.metadata.version(
                    "google-cloud-aiplatform"
                ),
                "google_adk_version": importlib.metadata.version("google-adk"),
            }
            print(json.dumps(preflight, indent=2))

            if observed_projects != {EXPECTED_PROJECT}:
                raise RuntimeError(
                    "Project mismatch: expected "
                    f"{EXPECTED_PROJECT}, observed {observed_projects}"
                )

            os.environ["GOOGLE_CLOUD_PROJECT"] = EXPECTED_PROJECT
            os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
            '''
        ),
        markdown(
            """
            ## 3. Deployment-safe Task 4 workflow

            The deployable root agent retains Task 4's Greeter and deterministic
            outer `SequentialAgent`. Search and Critique independently use ADK's
            built-in Google Search tool, then a Verified Draft stage creates the
            factual baseline. An inner `LoopAgent` runs Creativity followed by
            Refine for at most three cycles. Refine writes the durable answer on
            every cycle, so it is both the factuality gate and final author. The
            date is embedded at construction time so no hidden notebook state is
            required in the managed runtime.
            """
        ),
        code(
            r'''
            search_agent = Agent(
                name="search_agent",
                model=MODEL,
                description="Find current authoritative facts and write a draft.",
                instruction="""
                Use Google Search for every question and prefer official primary
                sources. Correct any misleading premise. Write a concise draft
                with VERIFIED FACTS, SOURCE NOTES, and DRAFT ANSWER sections.
                """,
                tools=[google_search],
                output_key="initial_answer",
            )

            critique_agent = Agent(
                name="critique_agent",
                model=MODEL,
                description="Independently fact-check and critique the draft.",
                instruction=f"""
                Runtime date: {CURRENT_DATE_UTC}

                Review this searched draft:
                --- INITIAL ANSWER ---
                {{initial_answer}}
                --- END INITIAL ANSWER ---

                Use Google Search to independently verify time-sensitive or
                disputed claims. Identify factual errors, unsupported claims,
                ambiguities, stale wording, and any missed correction. Name exact
                improvements for the Refine stage. Do not rewrite the answer or
                invent citation markers.
                """,
                tools=[google_search],
                output_key="critique",
            )

            verified_draft_agent = Agent(
                name="verified_draft_agent",
                model=MODEL,
                description="Consolidate the draft and critique into a factual baseline.",
                instruction="""
                Consolidate the searched draft and independent critique into a
                verified factual baseline.

                --- INITIAL ANSWER ---
                {initial_answer}
                --- END INITIAL ANSWER ---
                --- CRITIQUE ---
                {critique}
                --- END CRITIQUE ---

                Apply valid corrections, preserve dates, qualifiers, and source
                attributions, and remove unsupported claims. Do not add stylistic
                flourish. Return only the verified baseline.
                """,
                output_key="refined_answer",
            )

            creativity_agent = Agent(
                name="creativity_agent",
                model=MODEL,
                description=(
                    "Propose vivid, engaging language without changing verified facts."
                ),
                instruction="""
                Rewrite the current verified answer below as a creative candidate.

                --- CURRENT VERIFIED ANSWER ---
                {refined_answer}
                --- END CURRENT VERIFIED ANSWER ---

                Improve the opening, sentence rhythm, transitions, clarity, and
                memorability. Preserve every fact, date, qualifier, correction,
                and source attribution. Do not add claims, exaggerate, invent
                quotations, introduce unsupported metaphors, or mention the
                workflow. Return only the creative candidate.
                """,
                output_key="creative_candidate",
            )

            refine_agent = Agent(
                name="refine_agent",
                model=MODEL,
                description=(
                    "Fact-check the creative candidate and author the final answer."
                ),
                instruction="""
                Author the final audience-ready answer by comparing the creative
                candidate with the verified evidence and current factual baseline.

                --- INITIAL ANSWER ---
                {initial_answer}
                --- END INITIAL ANSWER ---
                --- CRITIQUE ---
                {critique}
                --- END CRITIQUE ---
                --- CURRENT VERIFIED ANSWER ---
                {refined_answer}
                --- END CURRENT VERIFIED ANSWER ---
                --- CREATIVE CANDIDATE ---
                {creative_candidate}
                --- END CREATIVE CANDIDATE ---

                Keep engaging language only where it is factually faithful.
                Restore any fact, date, qualifier, correction, or attribution that
                the candidate changed or omitted. Do not add unsupported claims.
                Return only the final answer; this output becomes the verified
                baseline for the next loop cycle.
                """,
                output_key="refined_answer",
            )

            creative_refinement_loop = LoopAgent(
                name="creative_refinement_loop",
                description=(
                    "Iterate Creativity then Refine, with Refine always authoring "
                    "the durable answer."
                ),
                max_iterations=3,
                sub_agents=[creativity_agent, refine_agent],
            )

            answer_team = SequentialAgent(
                name="answer_team",
                description=(
                    "Search, Critique, seed a verified draft, then iterate creative "
                    "refinement up to three times."
                ),
                sub_agents=[
                    search_agent,
                    critique_agent,
                    verified_draft_agent,
                    creative_refinement_loop,
                ],
            )

            greeter_agent = Agent(
                name="greeter",
                model=MODEL,
                description="Root agent that delegates questions to answer_team.",
                instruction="""
                For every nonempty factual or explanatory question, immediately
                transfer to answer_team. Never answer the question yourself and
                never skip the workflow.
                """,
                sub_agents=[answer_team],
            )

            architecture_evidence = {
                "source_notebook": "04_agent_workflow.ipynb",
                "root_agent": greeter_agent.name,
                "workflow_type": type(answer_team).__name__,
                "workflow_order": [agent.name for agent in answer_team.sub_agents],
                "loop_type": type(creative_refinement_loop).__name__,
                "loop_order": [
                    agent.name for agent in creative_refinement_loop.sub_agents
                ],
                "loop_max_iterations": creative_refinement_loop.max_iterations,
                "search_tool": "google_search",
                "critique_tool": "google_search",
                "state_handoffs": {
                    "draft": search_agent.output_key,
                    "critique": critique_agent.output_key,
                    "verified_seed": verified_draft_agent.output_key,
                    "creative_candidate": creativity_agent.output_key,
                    "final_refinement": refine_agent.output_key,
                },
            }
            print(json.dumps(architecture_evidence, indent=2))
            '''
        ),
        markdown(
            """
            ## 4. Local `AdkApp` test and boundary handling

            The same `AdkApp` object that will be deployed is tested locally. A
            fresh session is created explicitly. Invalid blank and oversized
            requests are rejected before a session or Gemini request is created.
            """
        ),
        code(
            r'''
            def validate_query(message: str) -> str:
                """Return normalized input or raise before the model is called."""
                normalized = " ".join(message.split())
                if not normalized:
                    raise ValueError("Please provide a nonempty question.")
                if len(normalized) > 1200:
                    raise ValueError("Question exceeds the 1,200-character limit.")
                return normalized


            def event_text(event: dict[str, Any]) -> str:
                """Extract text from one serialized ADK event."""
                parts = event.get("content", {}).get("parts", [])
                return " ".join(
                    part.get("text", "")
                    for part in parts
                    if isinstance(part, dict) and part.get("text")
                ).strip()


            def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
                """Return bounded, rubric-relevant evidence from ADK events."""
                authors = list(
                    dict.fromkeys(
                        str(event.get("author", ""))
                        for event in events
                        if event.get("author")
                    )
                )
                final_event = next(
                    (
                        event
                        for event in reversed(events)
                        if event_text(event)
                    ),
                    {},
                )
                final_text = event_text(final_event)
                return {
                    "event_count": len(events),
                    "authors": authors,
                    "author_counts": {
                        author: sum(
                            1
                            for event in events
                            if str(event.get("author", "")) == author
                        )
                        for author in authors
                    },
                    "final_author": final_event.get("author", ""),
                    "final_response": final_text[:2400],
                }


            adk_app = agent_engines.AdkApp(
                agent=greeter_agent,
                app_name="task5_verified_answer_agent",
                enable_tracing=False,
            )

            LOCAL_USER_ID = f"task5-local-{uuid.uuid4().hex[:12]}"
            local_session = await adk_app.async_create_session(user_id=LOCAL_USER_ID)
            local_prompt = validate_query(
                "When did NASA launch Voyager 1, and which planet did it fly past first?"
            )
            local_events: list[dict[str, Any]] = []
            async for event in adk_app.async_stream_query(
                user_id=LOCAL_USER_ID,
                session_id=local_session["id"],
                message=local_prompt,
            ):
                local_events.append(event)

            local_result = summarize_events(local_events)
            assert local_result["final_response"], local_result
            assert {
                "greeter", "search_agent", "critique_agent",
                "verified_draft_agent", "creativity_agent", "refine_agent",
            } <= set(local_result["authors"]), local_result
            assert local_result["author_counts"]["creativity_agent"] == 3, local_result
            assert local_result["author_counts"]["refine_agent"] == 3, local_result
            assert local_result["final_author"] == "refine_agent", local_result
            print(json.dumps({"session_id": local_session["id"], **local_result}, indent=2))

            rejected_cases = []
            for label, invalid_query in (
                ("blank", "   "),
                ("oversized", "x" * 1201),
            ):
                try:
                    validate_query(invalid_query)
                except ValueError as exc:
                    rejected_cases.append(
                        {"label": label, "accepted": False, "model_called": False,
                         "error": str(exc)}
                    )
            assert len(rejected_cases) == 2
            assert all(not item["model_called"] for item in rejected_cases)
            print(json.dumps(rejected_cases, indent=2))
            '''
        ),
        markdown(
            """
            ## 5. Agent Engine staging and client initialization

            In-memory deployment uses a Cloud Storage staging bucket in the same
            project and region. The code reuses the bucket when it already exists
            and creates only this narrowly named lab bucket when necessary.
            """
        ),
        code(
            """
            required_services = ["aiplatform.googleapis.com", "storage.googleapis.com"]
            service_status = {}
            for service in required_services:
                result = run_gcloud(
                    ["services", "list", "--enabled", f"--filter=config.name:{service}",
                     "--format=value(config.name)"]
                )
                service_status[service] = service in result.stdout.split()

            assert all(service_status.values()), service_status

            STAGING_BUCKET = f"gs://{EXPECTED_PROJECT}-agent-engine-staging"
            bucket_check = run_gcloud(["storage", "buckets", "describe", STAGING_BUCKET])
            bucket_created = False
            if bucket_check.returncode != 0:
                bucket_create = run_gcloud(
                    [
                        "storage", "buckets", "create", STAGING_BUCKET,
                        f"--project={EXPECTED_PROJECT}", f"--location={LOCATION}",
                        "--uniform-bucket-level-access",
                    ]
                )
                if bucket_create.returncode != 0:
                    raise RuntimeError(
                        "Staging bucket creation failed: "
                        + bucket_create.stderr.strip()[:600]
                    )
                bucket_created = True

            vertexai.init(
                project=EXPECTED_PROJECT,
                location=LOCATION,
                staging_bucket=STAGING_BUCKET,
            )

            DEPLOYMENT_REQUIREMENTS = [
                "google-cloud-aiplatform[agent_engines,adk]==1.164.0",
                "google-adk==1.39.0",
            ]
            deployment_preflight = {
                "services_enabled": service_status,
                "staging_bucket": STAGING_BUCKET,
                "bucket_created_this_run": bucket_created,
                "agent_engine_client_initialized": True,
                "runtime_requirements": DEPLOYMENT_REQUIREMENTS,
            }
            print(json.dumps(deployment_preflight, indent=2))
            """
        ),
        markdown(
            """
            ## 6. Deploy to Vertex AI Agent Engine

            This live step updates the previously graded Agent Engine in place. Its
            output must contain
            the fully qualified Agent Engine resource name used by later cells and
            by the grader.
            """
        ),
        code(
            """
            DISPLAY_NAME = "task5-creative-verified-answer-agent"
            EXISTING_RESOURCE_NAME = (
                "projects/571054353266/locations/us-central1/"
                "reasoningEngines/4093976020686733312"
            )
            remote_agent = agent_engines.update(
                EXISTING_RESOURCE_NAME,
                agent_engine=adk_app,
                requirements=DEPLOYMENT_REQUIREMENTS,
                display_name=DISPLAY_NAME,
                description=(
                    "Task 5 Search-Critique plus bounded Creativity-Refine loop."
                ),
            )

            RESOURCE_NAME = remote_agent.resource_name
            assert RESOURCE_NAME.startswith("projects/")
            assert "/locations/us-central1/reasoningEngines/" in RESOURCE_NAME
            print(
                json.dumps(
                    {
                        "deployment_status": "updated",
                        "display_name": DISPLAY_NAME,
                        "resource_name": RESOURCE_NAME,
                    },
                    indent=2,
                )
            )
            """
        ),
        markdown(
            """
            ## 7. Successful remote query

            The query below is sent to the deployed Agent Engine, not the local
            object. A new managed session is created first, and the saved response
            must include the search, critique, verified-draft, Creativity, and
            Refine authors. Creativity and Refine must each run three times, with
            Refine producing the nonempty final answer.
            """
        ),
        code(
            r'''
            REMOTE_USER_ID = f"task5-remote-{uuid.uuid4().hex[:12]}"
            remote_session = await remote_agent.async_create_session(
                user_id=REMOTE_USER_ID
            )
            remote_prompt = validate_query(
                "The first Earth Day was in 1975. Verify the year, correct the "
                "premise if needed, and identify the organizer most associated "
                "with launching it."
            )
            remote_events: list[dict[str, Any]] = []
            async for event in remote_agent.async_stream_query(
                user_id=REMOTE_USER_ID,
                session_id=remote_session["id"],
                message=remote_prompt,
            ):
                remote_events.append(event)

            remote_result = summarize_events(remote_events)
            remote_response = remote_result["final_response"]
            assert remote_response, remote_result
            assert {
                "greeter", "search_agent", "critique_agent",
                "verified_draft_agent", "creativity_agent", "refine_agent",
            } <= set(remote_result["authors"]), remote_result
            assert remote_result["author_counts"]["creativity_agent"] == 3, remote_result
            assert remote_result["author_counts"]["refine_agent"] == 3, remote_result
            assert remote_result["final_author"] == "refine_agent", remote_result
            assert "1970" in remote_response, remote_result
            print(
                json.dumps(
                    {
                        "resource_name": RESOURCE_NAME,
                        "remote_session_id": remote_session["id"],
                        "remote_query": remote_prompt,
                        **remote_result,
                    },
                    indent=2,
                )
            )
            '''
        ),
        markdown(
            """
            ## 8. Cleanup — intentionally retained for grading

            The cleanup call is present but guarded off. `DELETE_AGENT_ENGINE`
            remains `False`, so running all cells proves the deletion branch did
            not execute and the Agent Engine resource remains available to the
            instructor. Change it to `True` only after grading.
            """
        ),
        code(
            """
            DELETE_AGENT_ENGINE = False

            if DELETE_AGENT_ENGINE:
                agent_engines.delete(RESOURCE_NAME, force=True)
                print(f"Deleted Agent Engine: {RESOURCE_NAME}")
            else:
                print(
                    json.dumps(
                        {
                            "cleanup_executed": False,
                            "resource_retained_for_grading": RESOURCE_NAME,
                            "cleanup_method": "agent_engines.delete(RESOURCE_NAME, force=True)",
                        },
                        indent=2,
                    )
                )
            """
        ),
        markdown(
            """
            ## 9. Grading evidence

            These assertions connect the saved local, deployment, remote-query,
            boundary, and Cleanup outputs directly to the Task 5 criteria.
            """
        ),
        code(
            """
            grading_evidence = {
                "copied_from_task4": (
                    architecture_evidence["source_notebook"]
                    == "04_agent_workflow.ipynb"
                ),
                "google_adk_agent_created": greeter_agent.name == "greeter",
                "task4_workflow_extended_with_bounded_loop": architecture_evidence[
                    "workflow_order"
                ] == [
                    "search_agent", "critique_agent", "verified_draft_agent",
                    "creative_refinement_loop",
                ],
                "verified_draft_agent_created": (
                    verified_draft_agent.name == "verified_draft_agent"
                    and verified_draft_agent.output_key == "refined_answer"
                ),
                "creativity_agent_created": (
                    creativity_agent.name == "creativity_agent"
                    and creativity_agent.output_key == "creative_candidate"
                ),
                "loop_agent_created": (
                    architecture_evidence["loop_type"] == "LoopAgent"
                ),
                "loop_max_three_cycles": (
                    architecture_evidence["loop_max_iterations"] == 3
                ),
                "loop_order_creativity_then_refine": (
                    architecture_evidence["loop_order"]
                    == ["creativity_agent", "refine_agent"]
                ),
                "local_adk_app_test_passed": bool(local_result["final_response"]),
                "local_loop_ran_three_cycles": (
                    local_result["author_counts"]["creativity_agent"] == 3
                    and local_result["author_counts"]["refine_agent"] == 3
                ),
                "local_final_answer_is_refine_output": (
                    local_result["final_author"] == "refine_agent"
                ),
                "agent_engine_client_initialized": deployment_preflight[
                    "agent_engine_client_initialized"
                ],
                "explicit_runtime_requirements": DEPLOYMENT_REQUIREMENTS
                == [
                    "google-cloud-aiplatform[agent_engines,adk]==1.164.0",
                    "google-adk==1.39.0",
                ],
                "deployed_resource_name_saved": RESOURCE_NAME.startswith("projects/"),
                "remote_query_successful": bool(remote_response),
                "remote_workflow_authors_visible": {
                    "greeter", "search_agent", "critique_agent",
                    "verified_draft_agent", "creativity_agent", "refine_agent",
                }
                <= set(remote_result["authors"]),
                "remote_loop_ran_three_cycles": (
                    remote_result["author_counts"]["creativity_agent"] == 3
                    and remote_result["author_counts"]["refine_agent"] == 3
                ),
                "remote_final_answer_is_refine_output": (
                    remote_result["final_author"] == "refine_agent"
                ),
                "misleading_remote_premise_corrected": "1970" in remote_response,
                "invalid_input_stopped_before_model": all(
                    not item["accepted"] and not item["model_called"]
                    for item in rejected_cases
                ),
                "cleanup_code_present_but_not_run": not DELETE_AGENT_ENGINE,
                "resource_retained_for_grading": not DELETE_AGENT_ENGINE,
            }

            assert all(grading_evidence.values()), grading_evidence
            print(json.dumps(grading_evidence, indent=2))
            print(
                "TASK 5 COMPLETE: bounded Creativity-Refine loop update and "
                "remote query passed."
            )
            """
        ),
        markdown(
            """
            ## References

            - [Deploy an agent to Vertex AI Agent Engine](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/deploy)
            - [Vertex AI Agent Engine setup](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/set-up)
            - [Vertex AI Python `agent_engines` reference](https://docs.cloud.google.com/python/docs/reference/vertexai/latest/vertexai.agent_engines)
            - [Google ADK loop agents](https://adk.dev/agents/workflow-agents/loop-agents/)
            - [Google ADK sequential agents](https://adk.dev/agents/workflow-agents/sequential-agents/)
            """
        ),
    ]

    notebook["cells"] = cells
    metadata = notebook.setdefault("metadata", {})
    metadata.setdefault("colab", {})["name"] = "task 5.ipynb"
    TARGET.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Built {TARGET} with {len(cells)} cells from {SOURCE.name}.")


if __name__ == "__main__":
    main()
