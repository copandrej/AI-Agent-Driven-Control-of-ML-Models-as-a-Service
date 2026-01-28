import os
import subprocess
from dotenv import load_dotenv

# agentic stuff
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
load_dotenv()

# take from env for easy switching
VERBOSE = os.getenv("VERBOSE", "False").lower() == "true"
tracing = os.getenv("CREWAI_TRACING_ENABLED", "False").lower() == "true"
KEEP_MESSAGES = int(os.getenv("KEEP_MESSAGES", 7))
KEEP_FULL_OUTPUTS = int(os.getenv("KEEP_FULL_OUTPUTS", 2))

if VERBOSE:
    print("Configs:")
    print(f"Verbose: {VERBOSE}")
    print(f"Tracing: {tracing}")
    print(f"KEEP_MESSAGES: {KEEP_MESSAGES}")
    print(f"KEEP_FULL_OUTPUTS: {KEEP_FULL_OUTPUTS}")


class KubectlInput(BaseModel):
    command: str = Field(..., description="kubectl command without 'kubectl' prefix")


class KubectlTool(BaseTool):
    name: str = "kubectl"
    description: str = (
        "Execute kubectl commands on the cluster. "
        "Input: command without 'kubectl' prefix. "
        "Examples: 'get pods', 'apply -f file.yaml', 'create deployment name --image=nginx'"
    )
    args_schema: type[BaseModel] = KubectlInput
    
    def _run(self, command: str) -> str:
        try:
            result = subprocess.run(
                f"kubectl {command}", 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            return result.stdout if result.returncode == 0 else result.stderr
        except Exception as e:
            return f"Error: {str(e)}"


def main():
    # Setup LLM
    llm_provider = os.getenv("LLM_PROVIDER")
    if llm_provider == "ollama":
        os.environ["OLLAMA_API_BASE"] = os.getenv("OLLAMA_BASE_URL")
        llm = f"ollama/{os.getenv('OLLAMA_MODEL')}"
    elif llm_provider == "openrouter":
        os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
        llm = f"openrouter/{os.getenv('OPENROUTER_MODEL')}"
    else:
        print("Please configure the project in .env file first")
        os._exit(42)
    
    agent = Agent(
        role="Kubernetes Assistant",
        goal="Execute kubectl commands to help users manage their Kubernetes cluster",
        backstory=(
            "You are a Kubernetes assistant with kubectl access. "
            "WHEN TO USE KUBECTL TOOL:\n"
            "- User asks to CHECK/GET/LIST/SHOW cluster resources → use kubectl\n"
            "- User asks to CREATE/DEPLOY/RUN/SCALE/DELETE resources → use kubectl\n"
            "- User says something is incorrect → use kubectl to verify\n"
            "- User asks casual questions (who are you, thanks, hello) → answer naturally without tools\n\n"
            "RULES:\n"
            "- Never claim something worked without running kubectl to verify\n"
            "- If you need information, check first with kubectl\n"
            "- Never give instructions - execute commands yourself"
        ),
        tools=[KubectlTool()],
        llm=llm,
        verbose=VERBOSE,
        max_iter=15
    )
    
    print("Kubernetes Assistant (type 'exit' to quit)\n")
    
    conversation_history = []
    
    while True:
        query = input("You: ").strip()
        if query.lower() in ['exit', 'quit']:
            break
        
        # Build context with smart truncation
        context = ""
        if conversation_history:
            recent = conversation_history[-KEEP_MESSAGES:]
            context_parts = []
            
            for i, h in enumerate(recent):
                # For older messages, use heavily truncated version
                # For recent messages (last KEEP_FULL_OUTPUTS), use full version
                is_recent = i >= len(recent) - KEEP_FULL_OUTPUTS
                agent_response = h['agent_full'] if is_recent else h['agent_short']
                
                context_parts.append(f"User: {h['user']}\nYou: {agent_response}")
            
            context = "Recent context:\n" + "\n".join(context_parts) + "\n\n"
        
        task = Task(
            description=f"{context}Current request: {query}",
            expected_output="A helpful response - use kubectl tool when dealing with cluster resources",
            agent=agent
        )
        
        crew = Crew(
            agents=[agent], 
            tasks=[task],
            verbose=VERBOSE,
            memory=False
        )
        
        result = crew.kickoff()
        result_str = str(result)
        
        # Store both short and full versions
        conversation_history.append({
            "user": query,
            "agent_short": result_str[:100],  # Very short summary for old messages
            "agent_full": result_str[:500]    # Longer version for recent messages
        })
        
        # Trim history to prevent memory bloat
        if len(conversation_history) > KEEP_MESSAGES:
            conversation_history.pop(0)
        
        print(f"\nAgent: {result_str}\n")


if __name__ == "__main__":
    main()
