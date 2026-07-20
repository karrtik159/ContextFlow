"""
Memory Crew — background task to extract entities and update the knowledge graph.

Agents:
  - Entity_Extractor: Parses conversation transcripts for facts.
  - Graph_Updater: Persists extracted facts into Neo4j via Mem0.

Usage (fire-and-forget as a background task):
    MemoryCrew(user_id="...").crew().kickoff(inputs={"transcript": "..."})

As with SupportCrew, ``user_id`` is a constructor argument. MemoryStoreTool
*writes*, so an LLM-supplied scope here meant cross-tenant memory poisoning.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from agents.crews.tools.mem0_tool import MemoryStoreTool
from app.services.llm_provider import build_crewai_embedder, build_crewai_llm


@CrewBase
class MemoryCrew:
    """Background memory processor — extracts facts and updates the graph.

    Args:
        user_id: The tenant whose memory is being written. Required.
    """

    agents_config = "config/memory_agents.yaml"
    tasks_config = "config/memory_tasks.yaml"

    def __init__(self, user_id: str):
        if not user_id:
            raise ValueError("MemoryCrew requires a user_id — memory writes are tenant-scoped.")
        self.user_id = user_id
        # No super().__init__() — see the note in SupportCrew.__init__.

    @agent
    def entity_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config["entity_extractor"],
            verbose=True,
            max_iter=10,
            llm=build_crewai_llm(),
        )

    @agent
    def graph_updater(self) -> Agent:
        return Agent(
            config=self.agents_config["graph_updater"],
            verbose=True,
            max_iter=10,
            llm=build_crewai_llm(),
            tools=[MemoryStoreTool(user_id=self.user_id)],
        )

    @task
    def extract_entities(self) -> Task:
        return Task(config=self.tasks_config["extract_entities"])

    @task
    def update_graph(self) -> Task:
        return Task(
            config=self.tasks_config["update_graph"],
            context=[self.extract_entities()],
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
            max_execution_time=60,  # 1-min timeout (background job)
        )
