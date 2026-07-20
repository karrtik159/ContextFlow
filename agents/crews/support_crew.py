"""
Support Crew — synthesis only.

Retrieval is NOT done here any more. It runs deterministically in
``app/services/retrieval/`` and arrives as the pre-rendered ``context`` kickoff
input. What remains is a single tool-less agent making exactly one LLM call.

Usage:
    SupportCrew(user_id="...").crew().kickoff(
        inputs={"query": "...", "context": "<<<CONTEXT [1] ...>>>"}
    )

``user_id`` stays a constructor argument even though no tool consumes it now.
It scopes the crew instance and keeps the tenant boundary structural rather than
something callers can omit — and Phase 4's grounding checks will want it.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from app.services.llm_provider import build_crewai_embedder, build_crewai_llm


@CrewBase
class SupportCrew:
    """Synthesis crew — turns retrieved context into a cited answer.

    Args:
        user_id: The authenticated tenant scope. Required; there is no
            unscoped mode.
    """

    agents_config = "config/support_agents.yaml"
    tasks_config = "config/support_tasks.yaml"

    def __init__(self, user_id: str):
        if not user_id:
            raise ValueError("SupportCrew requires a user_id — RAG is tenant-scoped.")
        self.user_id = user_id
        # No super().__init__() call: @CrewBase applies a metaclass that
        # *rebuilds* the class, so the zero-arg super() cell would point at
        # the pre-rebuild class and raise TypeError. CrewBaseMeta.__call__
        # runs its own initialization after this returns.

    @agent
    def answer_synthesizer(self) -> Agent:
        return Agent(
            config=self.agents_config["answer_synthesizer"],
            verbose=True,
            # One shot. There is nothing to iterate towards: this agent has no
            # tools, so extra iterations can only re-word an answer at the cost
            # of another round-trip.
            max_iter=1,
            llm=build_crewai_llm(),
            # Still no tools. The synthesizer reads attacker-influenceable
            # retrieved content, so pairing it with a retrieval tool reopens the
            # stored-injection -> exfiltration loop that removing them closed.
            tools=[],
        )

    @task
    def synthesize_answer(self) -> Task:
        return Task(config=self.tasks_config["synthesize_answer"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            # memory=False deliberately. CrewAI's memory does its own embedding
            # and retrieval on every kickoff — additional calls that duplicate
            # the retrieval this pipeline just performed deterministically, and
            # that would defeat the phase's whole point of one LLM call per
            # knowledge query. Personalization comes from the memory arm of the
            # retrieval fan-out instead.
            memory=False,
            embedder=build_crewai_embedder(),
            cache=True,
            # Was 120s, which was a realistic p99 for a 10-iteration ReAct loop.
            # A single tool-less call that takes 30s has failed.
            max_execution_time=30,
        )
