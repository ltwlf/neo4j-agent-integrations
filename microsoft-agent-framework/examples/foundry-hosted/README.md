# Foundry-hosted multi-agent — Microsoft Agent Framework + Neo4j

Same multi-agent investment-research graph as [`../multi-agent/`](../multi-agent/) — a simple `SequentialBuilder` chain of specialist agents plus a final analyst — packaged as a [Foundry hosted agent](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agents) via [`agent-framework-foundry-hosting`](https://pypi.org/project/agent-framework-foundry-hosting/) (`ResponsesHostServer`). Deployed with the canonical [`azd ai agent init -m`](https://learn.microsoft.com/azure/foundry/agents/quickstarts/quickstart-hosted-agent) flow.

## Why host it?

Hosted agents take the same Agent Framework code you ran locally and put it on Foundry's managed runtime. From the [official concepts page](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agents):

- **Bring your own code** — Agent Framework, LangGraph, custom; the platform doesn't care.
- **Dedicated agent identity** — a Microsoft Entra ID is auto-created at deploy and used by the agent at runtime to call models, tools, and downstream Azure services. No managed-identity wiring.
- **Per-session VM-isolated sandboxes** — each session gets its own isolated sandbox; `$HOME` and `/files` persist across turns and idle periods, with compute deprovisioned after 15 minutes idle and restored on resume (up to 30-day session lifetime).
- **Versioning** — immutable agent versions with weighted traffic split for canary and blue-green rollouts.
- **Scale-to-zero** — Foundry handles container lifecycle, scaling, and Application Insights observability.
- **Foundry portal integration** — playground, version management, and traces, no extra wiring.

## Files in this folder

Flat layout matching the canonical [agent-framework hosted samples](https://github.com/microsoft/agent-framework/tree/main/python/samples/04-hosting/foundry-hosted-agents/responses):

| File | Purpose |
| --- | --- |
| `main.py` | The full hosted-agent definition: Neo4j `@tool` functions, specialist agent instructions, `SequentialBuilder`, and `ResponsesHostServer().run()`. Self-contained on purpose; [`../multi-agent/multi_agent_neo4j.py`](../multi-agent/multi_agent_neo4j.py) is a parallel near-identical file for local dev. |
| `requirements.txt` | Python deps (split-package install — see local example README) |
| `Dockerfile` | `python:3.12-slim`, exposes port 8088 |
| `.dockerignore` | Excludes `.azure/`, `.env`, `__pycache__/`, etc. |
| `agent.yaml` | Hosted-agent definition (protocol, resources, env vars) |
| `agent.manifest.yaml` | Template metadata + model resource — `azd ai agent init -m` reads this |
| `.env.example` | What to set locally for `python main.py` |
| `README.md` | This file |

## Quick demo

For this repo, the easiest path is the one Microsoft documents for local testing of hosted agents: scaffold the `azd` project, point it at the existing Foundry project, and run it locally with `azd ai agent run`. That keeps the sample easy to understand and avoids provisioning extra demo infrastructure like ACR and Application Insights unless you actually want a managed deployment.

### Prerequisites

```bash
azd ext install azure.ai.agents
az login
cd microsoft-foundry/infra && ./deploy.sh    # if you haven't already — provides the Foundry project
```

`microsoft-foundry/infra/deploy.sh` deploys to Sweden Central by default — a hosted-agents-supported region — so the same project can host this example.

### Run locally with the hosted-agent runtime

```bash
repo_root="$(git rev-parse --show-toplevel)"
manifest_path="$repo_root/microsoft-agent-framework/examples/foundry-hosted/agent.manifest.yaml"

# Reuse the shared Foundry deployment metadata written by
# microsoft-foundry/infra/deploy.sh
. "$repo_root/microsoft-foundry/.env"

PROJECT_ID="${FOUNDRY_PROJECT_ID:-/subscriptions/$AZURE_SUBSCRIPTION_ID/resourceGroups/$FOUNDRY_RESOURCE_GROUP/providers/Microsoft.CognitiveServices/accounts/$FOUNDRY_ACCOUNT_NAME/projects/$FOUNDRY_PROJECT_NAME}"
MODEL_DEPLOYMENT_NAME="$FOUNDRY_MODEL_DEPLOYMENT_NAME"

cd "$repo_root"
mkdir -p my-research-agent && cd my-research-agent

# 1. Scaffold the hosted-agent azd project against the existing Foundry
#    project + model.
azd ai agent init \
  -m "$manifest_path" \
  -p "$PROJECT_ID" \
  -d "$MODEL_DEPLOYMENT_NAME"

cd neo4j-research-agent-framework

# 2. Wire Neo4j (defaults connect to the public companies demo graph) +
#    the embedding deployment that microsoft-foundry/infra/ provisioned.
azd env set NEO4J_URI                       "neo4j+s://demo.neo4jlabs.com:7687"
azd env set NEO4J_DATABASE                  "companies"
azd env set NEO4J_USERNAME                  "companies"
azd env set NEO4J_PASSWORD                  "companies"
azd env set AZURE_TENANT_ID                 "$(az account show --query tenantId -o tsv)"
azd env set FOUNDRY_EMBEDDING_DEPLOYMENT_NAME "text-embedding-3-small"

# 3. Run the hosted-agent runtime locally.
azd ai agent run
```

In another terminal:

```bash
azd ai agent invoke --local --new-session \
  "Research Microsoft's position in the software industry. Gather company profile, recent news, and key relationships, then synthesize an investment outlook."
```

`azd ai agent invoke --local` is the canonical local test path for the hosted-agent runtime. `--new-session` keeps repeated demo runs isolated instead of reusing the prior conversation automatically.

For local runs, this sample now prefers `AzureCliCredential` when the Azure CLI is available, then falls back to `DefaultAzureCredential` for hosted deployment scenarios. Setting `AZURE_TENANT_ID` in the `azd` environment keeps local auth deterministic when your CLI can see multiple tenants.

You'll see a structured report — Executive Summary, Company Profile, Recent Developments, Network table, Risks & Outlook — with every `company_id` and `article_id` cited verbatim from the graph (real IDs like `EFhu1XwygPsKq_UjZtDFwXQ` and `ART11195006745`, not made-up placeholders).

## Deploy to Foundry (optional)

If you want a managed endpoint in Foundry after validating the demo locally, run:

```bash
azd up
```

from `my-research-agent/neo4j-research-agent-framework`.

That path follows the official hosted-agent flow: provision the small hosting resources for this agent, build the container remotely, push it to Azure Container Registry, and register the hosted agent version in the existing Foundry project.

### Tear down

```bash
# If you ran `azd up`, this removes the agent + any hosting resources
# provisioned by THIS folder. The shared Foundry account/project from
# microsoft-foundry/ stays alive — purge (`--purge`) would also delete
# the shared account, so leave it off.
azd down --no-prompt
```

To remove the entire shared deployment as well, run `azd down --purge --no-prompt` from `microsoft-foundry/infra/` afterwards.

## How it differs from `../multi-agent/`

Same agent graph, three differences in `main.py`:

1. **Tool decorator** — every Neo4j function is wrapped in `@tool(approval_mode="never_require")` with `Annotated[..., Field(description=...)]` parameter docs. Hosted agents default to requiring approval for tool calls; we opt out so the multi-agent flow runs unattended.
2. **Credential** — CLI-first locally, hosted-safe in Azure: the sample uses `AzureCliCredential` when `az` is available, chained to `DefaultAzureCredential()` for deployed runtime fallback.
3. **`default_options={"store": False}`** on each hosted agent — the hosting platform owns conversation history; don't double-persist on the OpenAI Responses side.

That's it. The multi-agent composition (`SequentialBuilder` specialists + analyst) and the anti-hallucination contract (JSON blocks per tool call, raw rows verbatim) are identical.
