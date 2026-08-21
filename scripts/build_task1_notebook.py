"""Build the complete Task 1 weather agent notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "01_weather_alerts_agent.ipynb"


def markdown(source: str) -> dict:
    """Return a notebook markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip() + "\n",
    }


def code(source: str) -> dict:
    """Return an unexecuted notebook code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


CELLS = [
    markdown(
        """
        # Challenge 1: weather alerts agent

        This notebook builds and tests a Google Agent Development Kit (ADK) agent
        that converts U.S. place names to coordinates with Google Maps, then gets
        observations, forecasts, and active alerts from the National Weather Service
        (NWS). The saved outputs provide the grading record.

        ## Goal

        - Create typed, documented Google Maps and NWS function tools.
        - Register both tools with a Gemini agent in Google ADK.
        - Return a current weather summary and any active alerts.
        - Prove the agent works for multiple U.S. cities.
        """
    ),
    markdown(
        """
        ## 1. Setup

        The notebook uses the ADK 1.x line required by the workshop. The Workbench
        service account supplies Vertex AI credentials. A restricted Google Maps key
        is loaded at runtime and never printed or saved in notebook output.
        """
    ),
    code(
        '''
        import importlib.util
        import subprocess
        import sys


        required_modules = ("google.adk", "requests")
        missing_modules = [
            module for module in required_modules if importlib.util.find_spec(module) is None
        ]
        if missing_modules:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--quiet",
                    "google-adk>=1.18,<2.0",
                    "requests>=2.32,<3",
                ],
                check=True,
            )
            print(f"Installed missing modules: {missing_modules}")
        else:
            print("Required Python modules are already installed.")
        '''
    ),
    code(
        '''
        from __future__ import annotations

        import importlib.metadata
        import json
        import os
        import subprocess
        import uuid
        from typing import Any

        import google.auth
        import requests


        EXPECTED_PROJECT = "qwiklabs-gcp-02-66b2cfb8579b"
        LOCATION = "us-central1"
        MODEL_LOCATION = "global"
        MODEL = "gemini-3.7-flash"


        def run_gcloud(arguments: list[str]) -> subprocess.CompletedProcess[str]:
            """Run a bounded gcloud command without printing credentials."""
            return subprocess.run(
                ["gcloud", *arguments],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )


        project_result = run_gcloud(["config", "get-value", "project"])
        detected_project = project_result.stdout.strip()
        _, adc_project = google.auth.default()
        observed_projects = {value for value in (detected_project, adc_project) if value}

        print(
            json.dumps(
                {
                    "expected_project": EXPECTED_PROJECT,
                    "gcloud_project": detected_project,
                    "adc_project": adc_project,
                    "location": LOCATION,
                    "model_location": MODEL_LOCATION,
                    "model": MODEL,
                    "google_adk_version": importlib.metadata.version("google-adk"),
                },
                indent=2,
            )
        )

        if observed_projects != {EXPECTED_PROJECT}:
            raise RuntimeError(
                f"Project mismatch: expected {EXPECTED_PROJECT}, observed {observed_projects}"
            )

        os.environ["GOOGLE_CLOUD_PROJECT"] = EXPECTED_PROJECT
        os.environ["GOOGLE_CLOUD_LOCATION"] = MODEL_LOCATION
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
        '''
    ),
    code(
        '''
        MAPS_KEY_DISPLAY_NAME = "task1-weather-geocoding-v2"


        def load_maps_api_key() -> str:
            """Load the Maps key from the environment or Google API Keys service."""
            environment_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
            if environment_key:
                return environment_key

            list_result = run_gcloud(
                [
                    "services",
                    "api-keys",
                    "list",
                    f"--filter=displayName={MAPS_KEY_DISPLAY_NAME}",
                    "--format=value(name)",
                ]
            )
            key_names = [line.strip() for line in list_result.stdout.splitlines() if line.strip()]
            if list_result.returncode or not key_names:
                raise RuntimeError(
                    "A restricted Google Maps key named "
                    f"{MAPS_KEY_DISPLAY_NAME!r} is required."
                )

            key_result = run_gcloud(
                [
                    "services",
                    "api-keys",
                    "get-key-string",
                    key_names[0],
                    "--format=value(keyString)",
                ]
            )
            key_string = key_result.stdout.strip()
            if key_result.returncode or not key_string:
                raise RuntimeError("The Maps key exists but its key string could not be loaded.")
            return key_string


        GOOGLE_MAPS_API_KEY = load_maps_api_key()
        print({"maps_credential_loaded": bool(GOOGLE_MAPS_API_KEY)})
        '''
    ),
    markdown(
        """
        ## 2. External API tools

        `geocode_place` restricts results to the United States and returns only the
        fields the agent needs. `get_weather` follows the NWS point metadata to the
        nearest observation station and forecast office, then checks active alerts for
        the same coordinates. Both tools return compact error objects instead of
        leaking request URLs or credentials through exceptions.
        """
    ),
    code(
        '''
        MAPS_GEOCODING_URL = "https://maps.googleapis.com/maps/api/geocode/json"
        NWS_API_ROOT = "https://api.weather.gov"
        REQUEST_TIMEOUT_SECONDS = 20
        NWS_HEADERS = {
            "Accept": "application/geo+json",
            "User-Agent": "task1-weather-agent/1.0 (Google Cloud skills workshop)",
        }


        class ExternalServiceError(RuntimeError):
            """Describe a safe external-service failure without including a secret URL."""


        def request_json(
            url: str,
            *,
            service_name: str,
            params: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
        ) -> dict[str, Any]:
            """Return JSON from an HTTP GET request or raise a sanitized error.

            Args:
                url: Service endpoint without user-facing logging.
                service_name: Safe name used in error messages.
                params: Optional query parameters.
                headers: Optional HTTP request headers.

            Returns:
                The decoded JSON object.

            Raises:
                ExternalServiceError: If the request or JSON decoding fails.
            """
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            except requests.RequestException as exc:
                raise ExternalServiceError(f"{service_name} request failed.") from exc

            if not response.ok:
                raise ExternalServiceError(
                    f"{service_name} returned HTTP {response.status_code}."
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise ExternalServiceError(f"{service_name} returned invalid JSON.") from exc
            if not isinstance(payload, dict):
                raise ExternalServiceError(f"{service_name} returned an unexpected payload.")
            return payload


        def geocode_place(place: str) -> dict[str, Any]:
            """Convert a U.S. place name to latitude and longitude with Google Maps.

            Args:
                place: A city, address, or named place in the United States.

            Returns:
                A compact dictionary with status, formatted address, coordinates, and
                place ID. Error results contain a safe message and no credential data.
            """
            normalized_place = place.strip()
            if not normalized_place:
                return {"status": "error", "message": "Place must not be empty."}

            try:
                payload = request_json(
                    MAPS_GEOCODING_URL,
                    service_name="Google Maps Geocoding API",
                    params={
                        "address": normalized_place,
                        "components": "country:US",
                        "key": GOOGLE_MAPS_API_KEY,
                    },
                )
            except ExternalServiceError as exc:
                return {"status": "error", "message": str(exc)}

            api_status = payload.get("status")
            results = payload.get("results") or []
            if api_status != "OK" or not results:
                safe_status = str(api_status or "UNKNOWN")
                return {
                    "status": "error",
                    "message": f"Google Maps found no usable result ({safe_status}).",
                }

            first_result = results[0]
            result_types = set(first_result.get("types", []))
            if first_result.get("partial_match") or result_types <= {"country", "political"}:
                return {
                    "status": "error",
                    "message": "Google Maps returned only a partial or country-level match.",
                }
            country_codes = {
                component.get("short_name")
                for component in first_result.get("address_components", [])
                if "country" in component.get("types", [])
            }
            if country_codes != {"US"}:
                return {"status": "error", "message": "The result is outside the United States."}

            location = first_result["geometry"]["location"]
            return {
                "status": "success",
                "query": normalized_place,
                "formatted_address": first_result.get("formatted_address"),
                "latitude": round(float(location["lat"]), 6),
                "longitude": round(float(location["lng"]), 6),
                "place_id": first_result.get("place_id"),
            }
        '''
    ),
    code(
        '''
        def celsius_to_fahrenheit(value: float | None) -> float | None:
            """Convert Celsius to Fahrenheit when a value is present."""
            return None if value is None else round((value * 9 / 5) + 32, 1)


        def meters_per_second_to_mph(value: float | None) -> float | None:
            """Convert meters per second to miles per hour when a value is present."""
            return None if value is None else round(value * 2.23694, 1)


        def measurement_value(properties: dict[str, Any], name: str) -> float | None:
            """Read a numeric NWS observation measurement when available."""
            measurement = properties.get(name) or {}
            value = measurement.get("value")
            return float(value) if isinstance(value, (int, float)) else None


        def get_weather(latitude: float, longitude: float) -> dict[str, Any]:
            """Get current NWS observations, forecast, and alerts for coordinates.

            Args:
                latitude: Latitude in decimal degrees from -90 through 90.
                longitude: Longitude in decimal degrees from -180 through 180.

            Returns:
                Current observation data, the nearest forecast period, and up to five
                active NWS alerts. Errors contain a safe, concise message.
            """
            if not -90 <= latitude <= 90:
                return {"status": "error", "message": "Latitude must be between -90 and 90."}
            if not -180 <= longitude <= 180:
                return {
                    "status": "error",
                    "message": "Longitude must be between -180 and 180.",
                }

            point = f"{latitude:.4f},{longitude:.4f}"
            try:
                point_payload = request_json(
                    f"{NWS_API_ROOT}/points/{point}",
                    service_name="NWS points service",
                    headers=NWS_HEADERS,
                )
                point_properties = point_payload["properties"]

                forecast_payload = request_json(
                    point_properties["forecast"],
                    service_name="NWS forecast service",
                    headers=NWS_HEADERS,
                )
                periods = forecast_payload.get("properties", {}).get("periods", [])
                if not periods:
                    raise ExternalServiceError("NWS forecast service returned no periods.")

                observation: dict[str, Any] = {"available": False}
                station_collection = request_json(
                    point_properties["observationStations"],
                    service_name="NWS station service",
                    headers=NWS_HEADERS,
                )
                station_urls = station_collection.get("observationStations", [])
                if station_urls:
                    latest_payload = request_json(
                        f"{station_urls[0]}/observations/latest",
                        service_name="NWS observation service",
                        headers=NWS_HEADERS,
                    )
                    latest = latest_payload.get("properties", {})
                    observation = {
                        "available": True,
                        "station": station_urls[0].rsplit("/", 1)[-1],
                        "timestamp": latest.get("timestamp"),
                        "description": latest.get("textDescription"),
                        "temperature_f": celsius_to_fahrenheit(
                            measurement_value(latest, "temperature")
                        ),
                        "humidity_percent": (
                            round(measurement_value(latest, "relativeHumidity"), 1)
                            if measurement_value(latest, "relativeHumidity") is not None
                            else None
                        ),
                        "wind_mph": meters_per_second_to_mph(
                            measurement_value(latest, "windSpeed")
                        ),
                    }

                alerts_payload = request_json(
                    f"{NWS_API_ROOT}/alerts/active",
                    service_name="NWS alerts service",
                    params={"point": point},
                    headers=NWS_HEADERS,
                )
                alerts = []
                for feature in alerts_payload.get("features", [])[:5]:
                    properties = feature.get("properties", {})
                    alerts.append(
                        {
                            "event": properties.get("event"),
                            "severity": properties.get("severity"),
                            "urgency": properties.get("urgency"),
                            "headline": properties.get("headline"),
                            "instruction": properties.get("instruction"),
                        }
                    )
            except (ExternalServiceError, KeyError, TypeError, ValueError) as exc:
                message = str(exc) if isinstance(exc, ExternalServiceError) else "NWS response was incomplete."
                return {"status": "error", "message": message}

            current_period = periods[0]
            alert_summary = (
                "; ".join(alert.get("event") or "Weather alert" for alert in alerts)
                if alerts
                else "No active NWS alerts."
            )
            return {
                "status": "success",
                "coordinates": {"latitude": latitude, "longitude": longitude},
                "location": {
                    "city": point_properties.get("relativeLocation", {})
                    .get("properties", {})
                    .get("city"),
                    "state": point_properties.get("relativeLocation", {})
                    .get("properties", {})
                    .get("state"),
                },
                "observation": observation,
                "forecast": {
                    "name": current_period.get("name"),
                    "temperature": current_period.get("temperature"),
                    "temperature_unit": current_period.get("temperatureUnit"),
                    "wind": f"{current_period.get('windSpeed')} {current_period.get('windDirection')}",
                    "short_forecast": current_period.get("shortForecast"),
                    "detailed_forecast": current_period.get("detailedForecast"),
                },
                "active_alert_count": len(alerts),
                "alert_summary": alert_summary,
                "alerts": alerts,
            }
        '''
    ),
    markdown(
        """
        ## 3. Deterministic checks

        These checks cover empty input and coordinate boundaries before any live test.
        They keep simple validation failures separate from network and model behavior.
        """
    ),
    code(
        '''
        assert geocode_place("   ") == {
            "status": "error",
            "message": "Place must not be empty.",
        }
        assert get_weather(90.01, 0)["status"] == "error"
        assert get_weather(0, -180.01)["status"] == "error"
        assert geocode_place.__annotations__["place"] == "str"
        assert get_weather.__annotations__["latitude"] == "float"
        assert geocode_place.__doc__ and get_weather.__doc__
        print("Deterministic validation checks: PASS")
        '''
    ),
    markdown(
        """
        ## 4. Live tool tests

        The test set spans the Northeast, Southeast, and Pacific Northwest. Each row
        must contain a Google Maps result and live NWS weather data before the agent
        test begins.
        """
    ),
    code(
        '''
        TEST_CITIES = ["New York, NY", "Miami, FL", "Seattle, WA"]
        direct_test_results: list[dict[str, Any]] = []

        for city in TEST_CITIES:
            geocode_result = geocode_place(city)
            assert geocode_result["status"] == "success", geocode_result

            weather_result = get_weather(
                geocode_result["latitude"],
                geocode_result["longitude"],
            )
            assert weather_result["status"] == "success", weather_result

            result = {
                "city": city,
                "formatted_address": geocode_result["formatted_address"],
                "coordinates": {
                    "latitude": geocode_result["latitude"],
                    "longitude": geocode_result["longitude"],
                },
                "observation": weather_result["observation"],
                "forecast": weather_result["forecast"],
                "active_alert_count": weather_result["active_alert_count"],
                "alert_summary": weather_result["alert_summary"],
            }
            direct_test_results.append(result)
            print(json.dumps(result, indent=2))

        print(f"Live external-tool tests: PASS ({len(direct_test_results)} cities)")
        '''
    ),
    code(
        '''
        invalid_place_result = geocode_place("This place should not exist 9z8y7x6w5v")
        assert invalid_place_result["status"] == "error", invalid_place_result
        print("Live no-result geocoding check: PASS")
        print(invalid_place_result)
        '''
    ),
    markdown(
        """
        ## 5. ADK weather agent

        The agent must call `geocode_place` first and pass its coordinates to
        `get_weather`. Its answer names the observation time, current conditions,
        forecast, and alert status. It must report tool errors instead of guessing.
        """
    ),
    code(
        '''
        from google.adk.agents import Agent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types


        weather_agent = Agent(
            name="realtime_weather_agent",
            model=MODEL,
            description="Gets live U.S. weather observations, forecasts, and NWS alerts.",
            instruction="""
            You are a U.S. weather agent. For every requested city:
            1. Call geocode_place with the user's location.
            2. If geocoding succeeds, call get_weather with the returned latitude and longitude.
            3. Give a short answer with the resolved location, observation timestamp and
               conditions when available, current forecast, and alert status.
            4. Put active NWS alerts first and state their severity and instructions.
            5. If a tool returns an error, explain the error plainly. Never invent weather.
            """,
            tools=[geocode_place, get_weather],
        )

        APP_NAME = "task1_weather_agent"
        USER_ID = "grader"
        session_service = InMemorySessionService()
        runner = Runner(
            agent=weather_agent,
            app_name=APP_NAME,
            session_service=session_service,
        )
        print(
            {
                "agent_name": weather_agent.name,
                "model": MODEL,
                "tools": [tool.__name__ for tool in (geocode_place, get_weather)],
            }
        )
        '''
    ),
    code(
        '''
        async def run_weather_agent(city: str) -> dict[str, Any]:
            """Run one ADK turn and return its visible tool trace and final answer."""
            session_id = f"weather-{uuid.uuid4().hex[:12]}"
            await session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
            )
            message = types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            f"Use both weather tools to report the current weather and "
                            f"active alerts for {city}."
                        )
                    )
                ],
            )

            tool_calls: list[dict[str, Any]] = []
            final_answer = ""
            async for event in runner.run_async(
                user_id=USER_ID,
                session_id=session_id,
                new_message=message,
            ):
                for call in event.get_function_calls():
                    tool_calls.append({"tool": call.name, "arguments": dict(call.args or {})})
                if event.is_final_response() and event.content:
                    final_answer = "".join(
                        part.text or "" for part in event.content.parts if part.text
                    ).strip()

            return {
                "city": city,
                "tool_calls": tool_calls,
                "final_answer": final_answer,
            }
        '''
    ),
    markdown(
        """
        ## 6. Live agent tests

        Each scenario uses a fresh ADK session. The saved trace proves that Gemini
        called both custom tools; the final text proves that the agent turned those
        responses into a weather summary or alert.
        """
    ),
    code(
        '''
        agent_test_results: list[dict[str, Any]] = []

        for city in TEST_CITIES:
            agent_result = await run_weather_agent(city)
            called_tools = {entry["tool"] for entry in agent_result["tool_calls"]}
            assert {"geocode_place", "get_weather"}.issubset(called_tools), agent_result
            assert agent_result["final_answer"], agent_result
            agent_test_results.append(agent_result)
            print(json.dumps(agent_result, indent=2))

        print(f"Live ADK agent tests: PASS ({len(agent_test_results)} cities)")
        '''
    ),
    markdown(
        """
        ## 7. Grading evidence

        The final cell maps each requirement to executed evidence. A passing result
        requires three successful direct API scenarios, three successful ADK scenarios,
        and visible calls to both tools for every city.
        """
    ),
    code(
        '''
        evidence = {
            "typed_and_documented_tools": bool(
                geocode_place.__annotations__
                and get_weather.__annotations__
                and geocode_place.__doc__
                and get_weather.__doc__
            ),
            "google_maps_geocoding_live": len(direct_test_results) == len(TEST_CITIES),
            "nws_weather_live": all(
                item["forecast"]["short_forecast"] for item in direct_test_results
            ),
            "multiple_us_cities": [item["city"] for item in direct_test_results],
            "adk_agent_created": weather_agent.name == "realtime_weather_agent",
            "agent_used_both_tools": all(
                {"geocode_place", "get_weather"}.issubset(
                    {call["tool"] for call in item["tool_calls"]}
                )
                for item in agent_test_results
            ),
            "weather_summaries_saved": all(
                item["final_answer"] for item in agent_test_results
            ),
        }

        required_boolean_checks = [
            value for value in evidence.values() if isinstance(value, bool)
        ]
        assert required_boolean_checks and all(required_boolean_checks), evidence
        print(json.dumps(evidence, indent=2))
        print("TASK 1 COMPLETE: all grading checks passed.")
        '''
    ),
    markdown(
        """
        ## References

        - [Google ADK function tools](https://adk.dev/tools/function-tools/)
        - [Google ADK sessions and runners](https://adk.dev/sessions/session/)
        - [Google Maps Geocoding requests](https://developers.google.com/maps/documentation/geocoding/guides-v3/requests-geocoding)
        - [National Weather Service API](https://www.weather.gov/documentation/services-web-api)
        """
    ),
]


NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {
            "display_name": "Python 3 (Local)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> int:
    """Write the notebook using the repository's deterministic JSON format."""
    OUTPUT.write_text(
        json.dumps(NOTEBOOK, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(CELLS)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
