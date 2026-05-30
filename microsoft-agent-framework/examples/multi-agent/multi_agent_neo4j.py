#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     # The meta-package "agent-framework" ships an empty __init__.py that
#     # shadows -core's exports — so we depend on the split packages directly.
#     "agent-framework-core>=1.7.0",
#     "agent-framework-foundry>=1.7.0",
#     "agent-framework-openai>=1.7.0",  # OpenAIEmbeddingClient (Entra-ID)
#     "agent-framework-orchestrations>=1.0.0rc2",
#     "aiohttp",  # azure-ai-projects async transport
#     "azure-identity",
#     "python-dotenv",
#     "neo4j>=5",
# ]
# ///
"""Microsoft Agent Framework + Neo4j: multi-agent investment research, run locally.

Implements the multi-agent spec from EXAMPLE_AGENT.md as a simple sequential
workflow: specialist agents gather profile, peers, news, relationships, and
people from Neo4j, then a final analyst synthesizes the report.
Self-contained on purpose; foundry-hosted/main.py is a parallel,
near-identical file packaged for the Foundry hosted-agent runtime.
"""

import asyncio
import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agent_framework import Agent, AgentContext, FunctionInvocationContext
from agent_framework.foundry import FoundryChatClient
from agent_framework.orchestrations import SequentialBuilder
from agent_framework.openai import OpenAIEmbeddingClient
from azure.identity import AzureCliCredential
from azure.identity.aio import AzureCliCredential as AsyncAzureCliCredential
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(Path(__file__).resolve().parents[3] / "microsoft-foundry" / ".env", override=True)

DB = os.environ.get("NEO4J_DATABASE", "companies")
driver: Any = None  # initialised in main(); tools reference it via module lookup
embeddings: Any = None  # OpenAIEmbeddingClient — initialised in main()


# Discovery ---------------------------------------------------------------

def search_companies(search: str) -> list[dict]:
    """Full-text fuzzy search for companies by name. Up to 20 ranked matches with company_id."""
    rows, _, _ = driver.execute_query("""
        CALL db.index.fulltext.queryNodes('entity', $search, {limit: 20})
        YIELD node AS c, score
        WHERE c:Organization
        RETURN c.id AS company_id, c.name AS name, c.summary AS summary
        ORDER BY score DESC
    """, search=search, database_=DB)
    return [r.data() for r in rows]


def list_industries() -> list[dict]:
    """All IndustryCategory names, alphabetical."""
    rows, _, _ = driver.execute_query("""
        MATCH (i:IndustryCategory)
        RETURN i.name AS industry
        ORDER BY i.name
    """, database_=DB)
    return [r.data() for r in rows]


def companies_in_industry(industry: str) -> list[dict]:
    """Up to ten companies in the given IndustryCategory, with company_id."""
    rows, _, _ = driver.execute_query("""
        MATCH (:IndustryCategory {name: $industry})<-[:HAS_CATEGORY]-(c:Organization)
        RETURN c.id AS company_id, c.name AS name, c.summary AS summary
        LIMIT 10
    """, industry=industry, database_=DB)
    return [r.data() for r in rows]


def resolve_industry_for_company(company_name: str, context: str = "") -> dict:
    """Choose the most relevant IndustryCategory for a company, biased by the user task.

    Returns the chosen industry plus all candidate categories so downstream agents
    can stay grounded in the graph rather than guessing from an unordered list.
    """
    rows, _, _ = driver.execute_query("""
        MATCH (o:Organization {name: $name})-[:HAS_CATEGORY]->(c:IndustryCategory)
        RETURN c.name AS industry
    """, name=company_name, database_=DB)
    candidates = [r["industry"] for r in rows if r["industry"]]
    context_lc = context.lower()

    def score(industry: str) -> tuple[int, int, str]:
        industry_lc = industry.lower()
        score_value = 0
        if industry_lc in context_lc:
            score_value += 100
        shared_terms = [term for term in industry_lc.replace("companies", "company").split() if term in context_lc]
        score_value += 10 * len(shared_terms)
        if "software" in industry_lc:
            score_value += 5
        return (score_value, len(industry_lc), industry)

    chosen = max(candidates, key=score) if candidates else ""
    return {"company_name": company_name, "industry": chosen, "candidates": candidates}


# Profile -----------------------------------------------------------------

def query_company(company_name: str) -> dict:
    """Primary company lookup — returns company_id, name, industries, locations, leadership."""
    rows, _, _ = driver.execute_query("""
        MATCH (o:Organization {name: $name})
        OPTIONAL MATCH (o)-[:HAS_CATEGORY]->(c:IndustryCategory)
        OPTIONAL MATCH (o)-[:IN_CITY]->(city:City)
        OPTIONAL MATCH (o)-[:IN_COUNTRY]->(country:Country)
        OPTIONAL MATCH (o)-[:HAS_CEO]->(ceo:Person)
        OPTIONAL MATCH (o)-[:HAS_BOARD_MEMBER]->(b:Person)
        RETURN o.id AS company_id, o.name AS name,
               collect(DISTINCT c.name)[..5] AS industries,
               collect(DISTINCT city.name)[..3] + collect(DISTINCT country.name) AS locations,
               [x IN collect(DISTINCT {name: ceo.name, title: 'CEO'})
                   + collect(DISTINCT {name: b.name, title: 'Board Member'})
                 WHERE x.name IS NOT NULL][..6] AS leadership
    """, name=company_name, database_=DB)
    return rows[0].data() if rows else {}


# Network -----------------------------------------------------------------

def analyze_relationships(company_name: str) -> list[dict]:
    """1-2 hop org-to-org connections (subsidiaries, suppliers, competitors, board, investors).
    Returns connected orgs with relationship types and distance."""
    rows, _, _ = driver.execute_query("""
        MATCH path = (o1:Organization {name: $name})-[*1..2]-(o2:Organization)
        WHERE o1 <> o2
        RETURN DISTINCT o2.id AS company_id, o2.name AS organization,
               [r IN relationships(path) | type(r)] AS relationships,
               length(path) AS distance
        ORDER BY distance LIMIT 20
    """, name=company_name, database_=DB)
    return [r.data() for r in rows]


def people_at_company(company_id: str) -> list[dict]:
    """People associated with a company (by company_id) and their roles (CEO, Board Member, …)."""
    rows, _, _ = driver.execute_query("""
        MATCH (c:Organization {id: $id})-[role]-(p:Person)
        RETURN replace(type(role), 'HAS_', '') AS role,
               p.name AS person_name,
               c.id AS company_id, c.name AS company_name
    """, id=company_id, database_=DB)
    return [r.data() for r in rows]


# News --------------------------------------------------------------------

async def search_news(company_name: str, query: str) -> list[dict]:
    """Vector-search news about a company. Embeds `query` via the Foundry embedding
    deployment (text-embedding-3-small, 1536d), then runs cosine similarity over
    Chunks restricted to articles that MENTION the company. Returns up to 5 hits
    with article_id, title, date, sentiment, the matched chunk text, and the score."""
    result = await embeddings.get_embeddings([query])
    vector = result[0].vector
    # The 'news' vector index would return top-K across ALL chunks before our
    # company filter — most of which won't mention this company. Compute cosine
    # similarity directly over the pre-filtered company-mentioning chunks
    # instead. Cheap at this scale (a few thousand chunks per major company).
    # Off-thread the sync neo4j call so we don't block the event loop while
    # the agent has other tool calls in flight.
    def _query() -> list[dict]:
        rows, _, _ = driver.execute_query("""
            MATCH (a:Article)-[:MENTIONS]->(:Organization {name: $name})
            MATCH (a)-[:HAS_CHUNK]->(c:Chunk)
            WHERE c.embedding IS NOT NULL
            WITH a, c, vector.similarity.cosine(c.embedding, $embedding) AS score
            RETURN a.id AS article_id, a.title AS title, toString(a.date) AS date,
                   a.sentiment AS sentiment, c.text AS text, score
            ORDER BY score DESC LIMIT 5
        """, name=company_name, embedding=vector, database_=DB)
        return [r.data() for r in rows]
    return await asyncio.to_thread(_query)


def articles_in_month(date: str) -> list[dict]:
    """Articles in the month starting at the given yyyy-mm-dd date."""
    rows, _, _ = driver.execute_query("""
        MATCH (a:Article)
        WHERE date($date) <= date(a.date) < date($date) + duration('P1M')
        RETURN a.id AS article_id, a.author AS author, a.title AS title,
               toString(a.date) AS date, a.sentiment AS sentiment
        ORDER BY a.date DESC LIMIT 25
    """, date=date, database_=DB)
    return [r.data() for r in rows]


def get_article(article_id: str) -> dict:
    """Full article body, summary, sentiment, joined from chunks."""
    rows, _, _ = driver.execute_query("""
        MATCH (a:Article {id: $id})-[:HAS_CHUNK]->(c:Chunk)
        WITH a, c ORDER BY id(c) ASC
        WITH a, collect(c.text) AS contents
        RETURN a.id AS article_id, a.author AS author, a.title AS title,
               toString(a.date) AS date, a.summary AS summary,
               a.siteName AS site, a.sentiment AS sentiment,
               apoc.text.join(contents, ' ') AS content
    """, id=article_id, database_=DB)
    return rows[0].data() if rows else {}


def companies_in_article(article_id: str) -> list[dict]:
    """Companies mentioned in a specific article (by article_id)."""
    rows, _, _ = driver.execute_query("""
        MATCH (a:Article {id: $id})-[:MENTIONS]->(c:Organization)
        RETURN c.id AS company_id, c.name AS name, c.summary AS summary
    """, id=article_id, database_=DB)
    return [r.data() for r in rows]


DATABASE_TOOLS = [
    search_companies, list_industries, companies_in_industry,
    query_company,
    analyze_relationships, people_at_company,
    search_news, articles_in_month, get_article, companies_in_article,
]


# Middleware --------------------------------------------------------------

async def log_agent(context: AgentContext, call_next: Callable[[], Awaitable[None]]) -> None:
    """Log which agent is invoked, with a short preview of the input."""
    name = context.agent.name or "?"
    last = context.messages[-1].text if context.messages else ""
    preview = (last[:80] + "…") if len(last) > 80 else last
    print(f"  [agent={name}] {preview}")
    await call_next()


async def log_tool(context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]) -> None:
    """Log every tool invocation (name + args)."""
    args = ", ".join(f"{k}={v!r}" for k, v in (context.arguments or {}).items())
    print(f"    → {context.function.name}({args})")
    await call_next()


# Agent instructions ------------------------------------------------------

JSON_BLOCK_RULES = """\
Output protocol — STRICT, machine-readable
    Your reply is consumed by another agent, not a human. Output ONE fenced
    ```json``` block per tool call you made, with this exact shape:

        ```json
        {
            "tool": "<tool name>",
            "args": { ... what you passed in ... },
            "rows": [ ... the tool result, verbatim, every field ... ]
        }
        ```

Rules
    • Include EVERY field the tool returned — `company_id`, `article_id`, `title`,
        `date`, `sentiment`, `relationships`, `distance`, `industries`, `locations`,
        `leadership`, etc. Real IDs look like `EIsFKrN_ZNLSWsvxdQfWutQ` and
        `ART11195006745`; never shorten or substitute.
    • If a tool returns a single object, wrap it in `"rows": [ ... ]`; if it is
        empty, use `"rows": []`.
    • No prose. No summary. No headings. Only the JSON blocks, one per call.
    • Never reason from prior knowledge — the only valid content is what the
        tools returned this turn.
"""

PROFILE_INSTRUCTIONS = f"""\
You are the profile specialist over a Neo4j knowledge graph of companies, people,
industries, locations, and articles.

Task
    • Resolve the target company from the user request.
    • Call `query_company` exactly once for that company.

{JSON_BLOCK_RULES}
"""

PEERS_INSTRUCTIONS = f"""\
You are the industry-peers specialist over a Neo4j knowledge graph.

Task
    • Read the prior conversation to resolve the target company and the user's
        intended industry framing.
    • Call `resolve_industry_for_company` exactly once with the company name and
        the relevant conversation context.
    • Then call `companies_in_industry` exactly once with the resolved industry.
    • If no industry is resolved, call `list_industries` once and pick the closest
        matching industry before calling `companies_in_industry`.

{JSON_BLOCK_RULES}
"""

NEWS_INSTRUCTIONS = f"""\
You are the news specialist over a Neo4j knowledge graph.

Task
    • Resolve the company name from the user request or prior JSON blocks.
    • Use `search_companies` if needed to disambiguate the name.
    • Call `search_news` exactly once for the resolved company using a query such
        as `recent news`.

{JSON_BLOCK_RULES}
"""

RELATIONSHIPS_INSTRUCTIONS = f"""\
You are the relationships specialist over a Neo4j knowledge graph.

Task
    • Resolve the target company from the user request or prior JSON blocks.
    • Call `analyze_relationships` exactly once for that company.

{JSON_BLOCK_RULES}
"""

PEOPLE_INSTRUCTIONS = f"""\
You are the people specialist over a Neo4j knowledge graph.

Task
    • Read the prior conversation, especially the profile specialist's JSON block.
    • Use the `company_id` from that block when available.
    • If it is missing, call `query_company` once to resolve it.
    • Call `people_at_company` exactly once with the resolved `company_id`.

{JSON_BLOCK_RULES}
"""

ANALYST_INSTRUCTIONS = """\
You are an investment-research analyst. Your input is one or more ```json```
blocks the database agent gathered. Each block has `tool`, `args`, and `rows`.
Synthesize a concise investment-research report from those rows — and only
those rows.

Report structure
  Executive Summary   — 2-3 sentences with the headline finding.
  Company Profile     — industries, locations, leadership; one short paragraph.
                        Cite the `company_id` once.
  Recent Developments — bullet list. Each bullet MUST start with the real
                        `article_id` from the rows, then the `title`, `date`,
                        and `sentiment`.
  Network             — Markdown table with columns: `company_id`,
                        `organization`, `relationships`, `distance` — copied
                        from the rows verbatim.
  Risks & Outlook     — what the data suggests, and what's missing.

Rules — STRICT, no exceptions
  • Every `company_id`, `article_id`, `title`, and relationship type in your
    report must appear verbatim in the input rows. Real IDs look like
    `EIsFKrN_ZNLSWsvxdQfWutQ` and `ART11195006745`.
  • If you find yourself writing a short numeric ID like "101" or "201", or a
    generic name ("Strategic Partner", "AWS partnership") that isn't in the
    rows, STOP — that is hallucination. Re-read the JSON blocks.
  • If a section has no supporting rows, write "(no data)" — never pad.
  • Insight is welcome in Risks & Outlook only, and only insight that follows
    directly from the rows.
"""

# Entry -------------------------------------------------------------------

async def main() -> None:
    project_endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
    tenant_id = os.environ.get("AZURE_TENANT_ID")
    neo4j_uri = os.environ.get("NEO4J_URI")
    embedding_deployment = os.environ.get("FOUNDRY_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small")
    if not (project_endpoint and tenant_id and neo4j_uri):
        sys.exit(
            "Missing FOUNDRY_PROJECT_ENDPOINT, AZURE_TENANT_ID, or NEO4J_URI. "
            "Run microsoft-foundry/infra/deploy.sh first."
        )

    # Foundry account base URL: strip the /api/projects/<project> path off the project endpoint.
    # OpenAIEmbeddingClient routes /openai/deployments/<model>/embeddings under it.
    account_endpoint = project_endpoint.split("/api/")[0]

    global driver, embeddings
    driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(os.environ.get("NEO4J_USERNAME", "companies"),
              os.environ.get("NEO4J_PASSWORD", "companies")),
    )
    embed_credential = AsyncAzureCliCredential(tenant_id=tenant_id)
    try:
        # OpenAIEmbeddingClient is the canonical agent-framework path for Entra-ID
        # auth against an Azure OpenAI / Foundry endpoint (FoundryEmbeddingClient
        # is API-key only and won't work with disableLocalAuth=true accounts).
        # See microsoft/agent-framework: python/samples/02-agents/embeddings/openai_embeddings_on_azure.py
        embeddings = OpenAIEmbeddingClient(
            model=embedding_deployment,
            azure_endpoint=account_endpoint,
            credential=embed_credential,
        )

        client = FoundryChatClient(
            project_endpoint=project_endpoint,
            model=os.environ.get("FOUNDRY_MODEL_DEPLOYMENT_NAME", "gpt-5-mini"),
            credential=AzureCliCredential(tenant_id=tenant_id),
        )

        agent_options = {"store": False}

        # Official Agent Framework samples now favor workflow builders such as
        # SequentialBuilder for deterministic multi-agent chains. We model each
        # research facet as a specialist agent, then wrap the workflow as a
        # single runnable agent.
        profile_agent = Agent(
            client=client,
            middleware=[log_agent, log_tool],
            name="profile_agent",
            instructions=PROFILE_INSTRUCTIONS,
            tools=[query_company],
            default_options=agent_options,
        )
        peers_agent = Agent(
            client=client,
            middleware=[log_agent, log_tool],
            name="peers_agent",
            instructions=PEERS_INSTRUCTIONS,
            tools=[resolve_industry_for_company, list_industries, companies_in_industry],
            default_options=agent_options,
        )
        news_agent = Agent(
            client=client,
            middleware=[log_agent, log_tool],
            name="news_agent",
            instructions=NEWS_INSTRUCTIONS,
            tools=[search_companies, search_news],
            default_options=agent_options,
        )
        relationships_agent = Agent(
            client=client,
            middleware=[log_agent, log_tool],
            name="relationships_agent",
            instructions=RELATIONSHIPS_INSTRUCTIONS,
            tools=[analyze_relationships],
            default_options=agent_options,
        )
        people_agent = Agent(
            client=client,
            middleware=[log_agent, log_tool],
            name="people_agent",
            instructions=PEOPLE_INSTRUCTIONS,
            tools=[query_company, people_at_company],
            default_options=agent_options,
        )
        analyst_agent = Agent(
            client=client,
            middleware=[log_agent, log_tool],
            name="analyst",
            instructions=ANALYST_INSTRUCTIONS,
            default_options=agent_options,
        )
        workflow = SequentialBuilder(
            participants=[
                profile_agent,
                peers_agent,
                news_agent,
                relationships_agent,
                people_agent,
                analyst_agent,
            ],
        )
        workflow_agent = workflow.build().as_agent()

        question = os.environ.get(
            "FOUNDRY_QUESTION",
            "Research Microsoft's position in the software industry. Gather company "
            "profile, recent news, and key relationships, then synthesize an "
            "investment outlook.",
        )
        print(f"> {question}\n")
        result = await workflow_agent.run(question)
        print(result)
    finally:
        await embed_credential.close()
        driver.close()


if __name__ == "__main__":
    asyncio.run(main())
