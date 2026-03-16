""" Sage Coordinator - Simplified for HomeBench """
import os
from typing import List, Any, Dict

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain.callbacks.base import BaseCallbackHandler

from .prompts import ACTIVE_REACT_COORDINATOR_PREFIX, ACTIVE_REACT_COORDINATOR_SUFFIX


class CommandExtractorCallback(BaseCallbackHandler):
    """Callback to extract executed commands from tool outputs"""
    
    def __init__(self):
        self.executed_commands = []
        self.errors = []
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes execution"""
        # Extract successful commands from execute_command tool
        if output.startswith("Success: Executed "):
            command = output.replace("Success: Executed ", "").strip()
            self.executed_commands.append(command)
        # Track errors
        elif output.startswith("Error:") or "error_input" in output.lower():
            self.errors.append(output)


class SAGECoordinator:
    """Simplified SAGE coordinator for HomeBench experiments"""

    def __init__(self, llm, tools: List[BaseTool]):
        """
        Initialize the SAGE coordinator
        
        Args:
            llm: Language model instance (e.g., ChatOpenAI)
            tools: List of tools available to the agent
        """
        self.llm = llm
        self.tools = tools
        
        # Create the prompt template for ReAct agent
        # Must include: {tools}, {tool_names}, {agent_scratchpad}, {input}
        prompt_template = ACTIVE_REACT_COORDINATOR_PREFIX + """

{tools}

""" + ACTIVE_REACT_COORDINATOR_SUFFIX
        
        # Create prompt with all required variables
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )
        
        # Create the agent
        self.agent = create_react_agent(llm, tools, self.prompt)
        
        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            verbose=False,  # Suppress detailed agent output
            handle_parsing_errors=True,
            max_iterations=10  # Allow more iterations for query + execute workflow
        )

    def execute_homebench(self, user_query: str, home_status_json: dict) -> Dict[str, Any]:
        """
        Execute a HomeBench test case using SAGE's tool-based workflow
        
        Args:
            user_query: User's command (e.g., "Turn on the light in the living room")
            home_status_json: JSON structure of the home environment for tools to query
            
        Returns:
            Dictionary with execution results including extracted commands
        """
        # Load the environment into all tools
        for tool in self.tools:
            if hasattr(tool, 'load_case'):
                tool.load_case(home_status_json)
        
        # Create callback to extract executed commands
        callback = CommandExtractorCallback()
        
        # Construct the prompt - NO environment description given upfront
        # The agent must use query_devices tool to discover devices
        full_prompt = f"""User Request: {user_query}

You are a smart home assistant. The user has given you a command.

IMPORTANT WORKFLOW:
1. First, use the 'query_devices' tool to discover what devices are available
2. Analyze which device(s) the user wants to control
3. Use 'get_device_state' if you need to check current state
4. Finally, use 'execute_command' to perform the action

Do NOT guess device names or room names. Always query first!"""

        try:
            # Execute the agent with callback
            result = self.agent_executor.invoke(
                {"input": full_prompt},
                config={"callbacks": [callback]}
            )
            
            natural_output = result.get("output", "")
            
            # Format output based on executed commands
            if callback.executed_commands:
                # Format as HomeBench expects: '''command1, command2, ...'''
                commands_str = ", ".join(callback.executed_commands)
                output = f"'''{commands_str}'''"
            elif callback.errors or "error" in natural_output.lower():
                # If there were errors but no successful commands
                output = "'''error_input'''"
            else:
                # If agent decided not to execute (e.g., device doesn't exist)
                # Check if it's a rejection/error case
                if any(word in natural_output.lower() for word in ["no", "not available", "cannot", "unable", "doesn't exist"]):
                    output = "'''error_input'''"
                else:
                    # Fallback to natural language output
                    output = natural_output
            
            return {
                "success": True,
                "output": output,
                "type": "execution",
                "executed_commands": callback.executed_commands,
                "natural_language_output": natural_output
            }
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Exception in agent execution: {str(e)}")
            print(f"Traceback: {error_details}")
            
            return {
                "success": False,
                "output": "'''error_input'''",
                "type": "error",
                "error_message": str(e),
                "error_details": error_details
            }

    def execute(self, command: str) -> str:
        """
        Simple execution method for compatibility
        
        Args:
            command: Command to execute
            
        Returns:
            Execution result as string
        """
        try:
            result = self.agent_executor.invoke({"input": command})
            return result.get("output", "")
        except Exception as e:
            return f"Error: {str(e)}"
