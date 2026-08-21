"""Create deterministic starter notebooks for the ADK skills workshop."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
PROJECT_ID = "qwiklabs-gcp-02-66b2cfb8579b"


@dataclass(frozen=True)
class Challenge:
    filename: str
    title: str
    points: int
    goal: str
    requirements: tuple[str, ...]
    implementation_sections: tuple[tuple[str, str], ...]
    test_cases: tuple[str, ...]
    deployment: bool = False


CHALLENGES = (
    Challenge(
        filename="01_weather_alerts_agent.ipynb",
        title="Challenge 1: Real-Time Weather Alerts Agent",
        points=15,
        goal="Create and test an ADK agent that uses external tools for current weather alerts.",
        requirements=(
            "Use the National Weather Service API for weather by latitude and longitude.",
            "Use the Google Maps Geocoding API to convert U.S. places to coordinates.",
            "Add both functions to an ADK weather agent with clear instructions.",
            "Provide a weather summary or alert from the current conditions.",
            "Test multiple U.S. cities and save the results.",
        ),
        implementation_sections=(
            (
                "External API tools",
                "Implement typed, documented geocoding and weather functions. Add timeouts, "
                "status checks, compact return objects, and useful error messages.",
            ),
            (
                "ADK agent",
                "Create the Gemini weather agent and instruct it to use the tools before "
                "writing a concise weather summary or alert.",
            ),
        ),
        test_cases=(
            "At least three U.S. cities from different regions",
            "A location with active or notable weather when available",
            "An invalid or ambiguous place",
            "A failed or mocked upstream API response",
            "Visible output that proves the ADK agent called its tools",
        ),
    ),
    Challenge(
        filename="02_callbacks.ipynb",
        title="Challenge 2: Enhancing Agents with Callbacks",
        points=15,
        goal="Add observable logging and input guardrails to the weather agent.",
        requirements=(
            "Log user prompts with an ADK callback.",
            "Log model responses with an ADK callback.",
            "Reject locations outside the United States before the model call.",
            "Reject malicious or mission-inappropriate input before the model call.",
            "Save allowed and blocked callback events.",
        ),
        implementation_sections=(
            (
                "Testable validation functions",
                "Keep location and malicious-input checks separate from ADK callback "
                "signatures so each rule can be tested directly.",
            ),
            (
                "ADK callbacks",
                "Add before-model prompt logging and validation plus after-model response "
                "logging. Return None only when normal execution should continue.",
            ),
        ),
        test_cases=(
            "A valid U.S. weather request",
            "A non-U.S. location",
            "A prompt-injection or malicious request",
            "An empty or malformed request",
            "Visible prompt and model-response log entries",
        ),
    ),
    Challenge(
        filename="03_multi_agent_system.ipynb",
        title="Challenge 3: Developing Multi-Agent Systems",
        points=15,
        goal="Coordinate weather and Google Search specialists with a root ADK agent.",
        requirements=(
            "Create a coordinating root agent.",
            "Create a weather agent that uses the custom weather tools.",
            "Create a search agent that uses ADK's built-in Google Search tool.",
            "Delegate requests to the correct specialist.",
            "Save events that prove sub-agent use.",
        ),
        implementation_sections=(
            (
                "Specialist agents",
                "Keep Google Search in its own specialist agent. Give the weather and search "
                "agents narrow descriptions that make routing choices observable.",
            ),
            (
                "Root coordinator",
                "Register the specialists as sub-agents or agent tools and give the root "
                "agent explicit delegation rules.",
            ),
        ),
        test_cases=(
            "A weather-only request",
            "A current-news search request",
            "A request that needs both weather and current information",
            "An unrelated request that the coordinator handles or refuses predictably",
            "Printed events naming the specialist that ran",
        ),
    ),
    Challenge(
        filename="04_agent_workflow.ipynb",
        title="Challenge 4: Programming an Agent Workflow",
        points=15,
        goal="Answer, critique, and refine a response through an ADK workflow.",
        requirements=(
            "Create Greeter, Search, Critique, and Refine agents.",
            "Use a SequentialAgent or LoopAgent for the answer team.",
            "Pass the initial answer and critique into the refinement stage.",
            "Return the refined answer to the user.",
            "Save events that show every workflow stage.",
        ),
        implementation_sections=(
            (
                "Workflow agents",
                "Define distinct Greeter, Search, Critique, and Refine responsibilities. "
                "Use session state or output keys to pass results between stages.",
            ),
            (
                "Workflow assembly",
                "Build a deterministic SequentialAgent or a bounded LoopAgent. If looping, "
                "include an explicit stop condition and maximum iteration count.",
            ),
        ),
        test_cases=(
            "A factual question that benefits from current Google Search results",
            "A question whose first answer needs a clear correction",
            "Printed events for Search, Critique, and Refine",
            "A comparison of the initial and final answer",
        ),
    ),
    Challenge(
        filename="05_deploy_agent.ipynb",
        title="Bonus Challenge 5: Deploying an Agent",
        points=10,
        goal="Deploy an ADK application to Vertex AI Agent Engine and query it remotely.",
        requirements=(
            "Create and test an ADK agent locally.",
            "Initialize the current Vertex AI Agent Engine client.",
            "Deploy the agent with explicit runtime requirements.",
            "Save the deployed resource name.",
            "Save a successful remote query response.",
        ),
        implementation_sections=(
            (
                "Local application",
                "Wrap the tested agent in agent_engines.AdkApp and run a local query before "
                "creating any cloud resource.",
            ),
            (
                "Agent Engine deployment",
                "Create the remote Agent Engine only after the project check passes. Record "
                "the resource name and requirements used for deployment.",
            ),
        ),
        test_cases=(
            "A successful local query",
            "A successful remote query",
            "The remote Agent Engine resource name",
            "A documented failure if Qwiklabs blocks deployment permissions",
        ),
        deployment=True,
    ),
    Challenge(
        filename="06_readynow_emergency_assistant.ipynb",
        title="Challenge 6: ReadyNow Emergency Preparedness Assistant",
        points=40,
        goal="Build, validate, deploy, and test the complete ReadyNow multi-agent system.",
        requirements=(
            "Render an architecture diagram in the notebook.",
            "Create a root agent plus weather, Google Search, Google Maps routes, and safety agents.",
            "Provide real-time weather, news, routes to safety, and preparedness information.",
            "Use a sequential workflow to validate and refine responses.",
            "Log all user-agent interactions with callbacks.",
            "Reject input outside the emergency-preparedness mission.",
            "Deploy to Vertex AI Agent Engine and save local and remote tests.",
        ),
        implementation_sections=(
            (
                "Architecture diagram",
                "Render a diagram that shows the root coordinator, specialists, validation "
                "and refinement workflow, external APIs, logging, and Agent Engine.",
            ),
            (
                "Tools and specialist agents",
                "Implement weather, Google Search, Google Maps Routes, and safety specialists "
                "with typed tools and explicit error handling.",
            ),
            (
                "Guardrails and refinement",
                "Add callbacks for interaction logging and mission validation. Pass draft "
                "responses through validation, critique, and refinement.",
            ),
            (
                "Root agent and deployment",
                "Describe ReadyNow's scope, coordinate specialists, test locally, deploy to "
                "Agent Engine, and query the deployed application.",
            ),
        ),
        test_cases=(
            "Severe-weather summary and alert",
            "Current emergency news",
            "A route to a plausible safe destination",
            "A preparedness or safety question",
            "A request that needs more location detail",
            "An out-of-mission or malicious request that is refused",
            "An upstream API failure with a safe, useful response",
            "A successful local query and a successful deployed query",
        ),
        deployment=True,
    ),
)


def markdown(source: str) -> dict:
    """Return a notebook markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code(source: str) -> dict:
    """Return an unexecuted notebook code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


def checklist(items: tuple[str, ...]) -> str:
    """Format grading requirements as unchecked Markdown items."""
    return "\n".join(f"- [ ] {item}" for item in items)


def common_cells(challenge: Challenge) -> list[dict]:
    """Build the shared title, installation, and project-preflight cells."""
    return [
        markdown(
            f"""
            # {challenge.title}

            **Points:** {challenge.points}

            **Goal:** {challenge.goal}

            ## Grading checklist

            {checklist(challenge.requirements)}
            """
        ),
        markdown(
            """
            ## 1. Runtime setup

            Run this section first in a fresh Colab Enterprise runtime. Restart the
            runtime after dependency changes if an import reports a stale version.
            """
        ),
        code(
            """
            # Keep the workshop on the ADK 1.x compatibility line.
            %pip install --quiet \
                "google-adk~=1.0" \
                "google-cloud-aiplatform[agent_engines,adk]>=1.112.0" \
                "requests>=2.32,<3"
            """
        ),
        code(
            f"""
            import os
            import shutil
            import subprocess

            import google.auth

            PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "{PROJECT_ID}")
            LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            MODEL_LOCATION = os.getenv("ADK_MODEL_LOCATION", "global")
            MODEL = os.getenv("ADK_MODEL", "gemini-3.7-flash")

            # Keep resource deployment regional while Gemini uses its supported endpoint.
            os.environ["GOOGLE_CLOUD_LOCATION"] = MODEL_LOCATION

            # Colab Enterprise supports browser-based user authentication.
            try:
                from google.colab import auth as colab_auth
            except ImportError:
                colab_auth = None

            if colab_auth is not None:
                colab_auth.authenticate_user(project_id=PROJECT_ID)

            credentials, detected_project = google.auth.default()
            del credentials  # Never print or inspect credential contents.

            gcloud_project = ""
            if shutil.which("gcloud"):
                result = subprocess.run(
                    ["gcloud", "config", "get-value", "project"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                gcloud_project = result.stdout.strip()

            print({{
                "expected_project": PROJECT_ID,
                "detected_adc_project": detected_project,
                "gcloud_project": gcloud_project,
                "location": LOCATION,
                "model_location": MODEL_LOCATION,
                "model": MODEL,
            }})

            observed_projects = {{p for p in (detected_project, gcloud_project) if p}}
            if observed_projects and observed_projects != {{PROJECT_ID}}:
                raise RuntimeError(
                    f"Project mismatch: expected {{PROJECT_ID}}, observed {{observed_projects}}"
                )
            """
        ),
        code(
            """
            # Check secret names without printing their values.
            secret_names = ["GOOGLE_MAPS_API_KEY"]
            print({name: bool(os.getenv(name)) for name in secret_names})
            """
        ),
    ]


def implementation_cells(challenge: Challenge) -> list[dict]:
    """Build challenge-specific implementation placeholders."""
    cells = [markdown("## 2. Implementation")]
    for index, (heading, guidance) in enumerate(challenge.implementation_sections, start=1):
        cells.extend(
            [
                markdown(f"### 2.{index} {heading}\n\n{guidance}"),
                code(
                    f"""
                    # TODO: Implement {heading.lower()} here.
                    # Keep functions small and include type hints and useful docstrings.
                    # Do not print credentials, raw request headers, or authentication objects.
                    """
                ),
            ]
        )
    return cells


def test_cells(challenge: Challenge) -> list[dict]:
    """Build the test matrix and evidence placeholders."""
    scenarios = "\n".join(f"- [ ] {case}" for case in challenge.test_cases)
    cells = [
        markdown(
            f"""
            ## 3. Local tests

            Run each scenario and keep concise output that identifies the tool, callback,
            sub-agent, or workflow stage responsible for the result.

            {scenarios}
            """
        ),
        code(
            """
            # TODO: Add deterministic unit tests and live end-to-end scenarios.
            # Print compact event records that prove routing and tool behavior.
            test_results = []
            print(test_results)
            """
        ),
    ]
    if challenge.deployment:
        cells.extend(
            [
                markdown(
                    """
                    ## 4. Deployment test

                    Deploy only after local tests and the project preflight pass. Save the
                    Agent Engine resource name and at least one successful remote response.
                    """
                ),
                code(
                    """
                    # TODO: Initialize the Vertex AI client, wrap the tested ADK agent in
                    # agent_engines.AdkApp, deploy it, and run a remote query.
                    # Print the resource name, not credentials or request headers.
                    """
                ),
                markdown(
                    """
                    ### Cleanup

                    Add cleanup code and document the resource it deletes. Leave cleanup
                    unexecuted until the instructor no longer needs the deployed resource.
                    """
                ),
                code(
                    """
                    # TODO: Add an explicit Agent Engine deletion command here.
                    # Do not run it before grading.
                    """
                ),
            ]
        )
    return cells


def evidence_cells(challenge: Challenge) -> list[dict]:
    """Build the final requirement-to-output mapping."""
    section_number = 5 if challenge.deployment else 4
    return [
        markdown(
            f"""
            ## {section_number}. Grading evidence

            Complete this section after a clean Run all.

            {checklist(challenge.requirements)}

            **Evidence map**

            - TODO: Link each requirement to the cell output that proves it.
            - TODO: Record any live limitation, exact error, and smallest next action.
            - TODO: Confirm that code, markdown, and outputs contain no credentials.
            """
        )
    ]


def build_notebook(challenge: Challenge) -> dict:
    """Return a complete notebook document for one challenge."""
    cells = common_cells(challenge)
    cells.extend(implementation_cells(challenge))
    cells.extend(test_cells(challenge))
    cells.extend(evidence_cells(challenge))
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": challenge.filename, "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing notebooks. This discards notebook work.",
    )
    return parser.parse_args()


def main() -> int:
    """Create missing notebooks and refuse accidental overwrites."""
    args = parse_args()
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    blocked = [
        challenge.filename
        for challenge in CHALLENGES
        if (NOTEBOOK_DIR / challenge.filename).exists() and not args.force
    ]
    if blocked:
        print("Refusing to overwrite existing notebooks:")
        for filename in blocked:
            print(f"  {filename}")
        print("Use --force only when overwriting is intentional.")
        return 2

    for challenge in CHALLENGES:
        path = NOTEBOOK_DIR / challenge.filename
        path.write_text(
            json.dumps(build_notebook(challenge), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Created {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
