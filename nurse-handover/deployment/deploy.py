# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deployment script for Nurse Handover Agent"""

import os
import sys
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vertexai
from dotenv import load_dotenv
from vertexai import agent_engines
# from vertexai.preview.reasoning_engines import AdkApp
from vertexai.agent_engines import AdkApp

from nurse_handover.agent import root_agent


def create() -> None:
    """Creates a reasoning engine for the Nurse Handover workflow."""
    adk_app = AdkApp(agent=root_agent, enable_tracing=True)

    remote_agent = agent_engines.create(
        adk_app,
        display_name=root_agent.name,
        requirements=[
            "google-adk>=1.31.1",
            "google-cloud-aiplatform[agent-engines]>=1.148.1",
            "google-genai>=1.73.1",
            "pydantic>=2.12.5",
            "python-dotenv>=1.2.2",
            "pandas>=2.2.3,<3.0.0",
            "google-cloud-logging>=3.15.0",
        ],
        extra_packages=["./nurse_handover"],
    )
    print(f"Created remote agent: {remote_agent.resource_name}")


def delete(resource_id: str) -> None:
    """Deletes the specified reasoning engine."""
    remote_agent = agent_engines.get(resource_id)
    remote_agent.delete(force=True)
    print(f"Deleted remote agent: {resource_id}")


def list_agents() -> None:
    """Lists all reasoning engines in the project/location."""
    remote_agents = agent_engines.list()
    template = """
{agent.name} ("{agent.display_name}")
- Create time: {agent.create_time}
- Update time: {agent.update_time}
"""
    remote_agents_string = "\n".join(
        template.format(agent=agent) for agent in remote_agents
    )
    print(f"All remote agents:\n{remote_agents_string}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Nurse Handover Agent Deployment")
    parser.add_argument("--project_id", help="GCP project ID.")
    parser.add_argument("--location", help="GCP location.")
    parser.add_argument("--bucket", help="GCP bucket (staging).")
    parser.add_argument("--resource_id", help="ReasoningEngine resource ID.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="List all agents.")
    group.add_argument("--create", action="store_true", help="Creates a new agent on Reasoning Engine.")
    group.add_argument("--delete", action="store_true", help="Deletes an existing agent.")

    args = parser.parse_args()

    # Priority: Flag -> Env -> None
    project_id = args.project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = (
        args.location or os.getenv("GOOGLE_CLOUD_LOCATION") or "us-east1"
    )
    # Using the existing bucket from .env
    bucket = args.bucket or os.getenv("AE_DEPLOYMENT_BUCKET")

    print(f"PROJECT: {project_id}")
    print(f"LOCATION: {location}")
    print(f"STAGING BUCKET: {bucket}")

    if not project_id:
        print(
            "Error: Missing GOOGLE_CLOUD_PROJECT. Use --project_id or set in .env"
        )
        return
    if not bucket:
        print(
            "Error: Missing AE_DEPLOYMENT_BUCKET. Use --bucket or set in .env"
        )
        return

    vertexai.init(
        project=project_id,
        location=location,
        staging_bucket=f"gs://{bucket}",
    )

    if args.list:
        list_agents()
    elif args.create:
        create()
    elif args.delete:
        if not args.resource_id:
            print("Error: --resource_id is required for delete")
            return
        delete(args.resource_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
