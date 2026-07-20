"""
Support Crew — handles complex knowledge queries via Hybrid Graph-Vector RAG.

Agents:
  - Context_Gatherer: Queries pgvector + Neo4j + user memories in parallel.
  - Answer_Synthesizer: Generates a personalized response from context.

Usage:
    result = SupportCrew(user_id="...").crew().kickoff(inputs={"query": "..."})

The ``user_id`` is a constructor argument, not a kickoff input. Every
retrieval tool is built bound to it, so tenant scope is structural rather
than something the LLM is asked to pass along in prompt text.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from agents.crews.tools.graph_search_tool import GraphSearchTool
from agents.crews.tools.mem0_tool import MemorySearchTool
from agents.crews.tools.vector_search_tool import VectorSearchTool
from app.services.llm_provider import build_crewai_embedder, build_crewai_llm


@CrewBase
class SupportCrew:
    """Hybrid RAG crew — retrieves context and synthesizes answers.

    Args:
        user_id: The authenticated tenant scope. Required — there is no
            unscoped mode, because every store this crew reads (pgvector
            messages, the Neo4j subgraph, Mem0 memories) is user-owned.
    """

    agents_config = "config/support_agents.yaml"
    tasks_config = "config/support_tasks.yaml"

    def __init__(self, user_id: str):
        if not user_id:
            raise ValueError("SupportCrew requires a user_id — retrieval is tenant-scoped.")
        self.user_id = user_id
        # No super().__init__() call: @CrewBase applies a metaclass that
        # *rebuilds* the class, so the zero-arg super() cell would point at
        # the pre-rebuild class and raise TypeError. CrewBaseMeta.__call__
        # runs its own initialization (config load, agent/task mapping) after
        # this returns, which is why self.user_id is already set when the
        # @agent methods construct their tools.

    @agent
    def context_gatherer(self) -> Agent:
        return Agent(
            config=self.agents_config["context_gatherer"],
            verbose=True,
            max_iter=10,
            llm=build_crewai_llm(),
            tools=[
                VectorSearchTool(user_id=self.user_id),
                GraphSearchTool(user_id=self.user_id),
                MemorySearchTool(user_id=self.user_id),
            ],
        )

    @agent
    def answer_synthesizer(self) -> Agent:
        return Agent(
            config=self.agents_config["answer_synthesizer"],
            verbose=True,
            max_iter=10,
            llm=build_crewai_llm(),
            # No tools. The synthesizer reads retrieved content — which is
            # attacker-influenceable — so giving it a retrieval tool closed a
            # stored-injection -> exfiltration loop. Context arrives via the
            # task dependency on retrieve_context instead.
            tools=[],
        )

    @task
    def retrieve_context(self) -> Task:
        return Task(config=self.tasks_config["retrieve_context"])

    @task
    def synthesize_answer(self) -> Task:
        return Task(
            config=self.tasks_config["synthesize_answer"],
            context=[self.retrieve_context()],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=True,
            embedder=build_crewai_embedder(),
            cache=True,
            max_execution_time=120,  # 2-min hard timeout
        )
