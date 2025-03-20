from ..common.logger import LOGGER 
from .agent_engines import Engines
from .agent_state import AgentState
import sys
import io


########################################################################################
# Agent Function: start_agent
# Description: This function is used to start the agent-assist
########################################################################################
def start_agent(state: AgentState):
    LOGGER.info("Starting the agent-assist")
    state['state'] = 'start'
    return state


# It is not an agent, it is a redirector
########################################################################################
# Agent Function: check_user_qry
# Description: This function is used to check the user query
# Note: This function is not an agent, it is a redirector. It will be used directly as a function in the agent pipeline.
########################################################################################
def check_user_qry(state: AgentState):
    LOGGER.info("Checking user query")
    state['state'] = 'check_user_qry'   

    result = Engines.get_check_user_qry_engine().run(
        [
            f"You are an AI assistant. Your task is to analyze the user query and classify it into python, sql or other.",
            f"The user query is: {state['results']['user']}"
            f"Now, classify the user query into python, sql or other. Use the tool 'QueryClassfication' for this. Strictly follow the the arguments and return the result."
        ]
    )[0]

    if result['is_python']:
        return "python"
    elif result['is_sql']:
        return "sql"
    else:
        return "other"


########################################################################################
# Agent Function: write_python_code_agent
# Description: This function is used to write python code
########################################################################################
def write_python_code_agent(state: AgentState):
    LOGGER.info("Writing python code")
    state['state'] = 'write_python_code'


    result = Engines.get_generate_python_code_engine().run(
        [
            f"You are an AI assistant. Your task is to generate python code.",
            f"User's query is: {state['results']['user']}"
            f"Now, generate the full python code. Use the tool 'PythonCode' for this. Strictly follow the the arguments and return the result."
        ]
    )[0]

    state['results']["write_python_code"] = result['code']
    return state




########################################################################################
# Agent Function: python_code_runner_agent
# Description: This function is used to run the python code
########################################################################################
def python_code_runner_agent(state: AgentState):
    LOGGER.info("Running the python code")
    state['state'] = 'python_code_runner'
    code_str = state['results']['write_python_code']

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    local_scope = {}
    try:
        exec(code_str, {}, local_scope)
        output = sys.stdout.getvalue()
        LOGGER.info(f"Code executed successfully: {output}")
    except Exception as e:
        LOGGER.error(f"Error during execution: {e}")
        output = f"Error during execution: {e}"
    finally:
        sys.stdout = old_stdout  # Reset stdout
    
    state['results']["python_code_output"] = output
    return state


########################################################################################
# Agent Function: write_sql_code_agent
# Description: This function is used to write sql code
########################################################################################
def write_sql_code_agent(state: AgentState):
    LOGGER.info("Writing sql code")
    state['state'] = 'write_sql_code'
    
    result = Engines.get_generate_sql_code_engine().run(
        [
            f"You are an AI assistant. Your task is to generate sql code.",
            f"User's query is: {state['results']['user']}"
            f"Now, generate the sql code. Use the tool 'SqlCode' for this. Strictly follow the the arguments and return the result."
        ]
    )[0]

    state['results']["write_sql_code"] = result['code']
    return state

########################################################################################
# Agent Function: sql_code_runner_agent
# Description: This function is used to run the sql code
########################################################################################
def sql_code_runner_agent(state: AgentState):
    LOGGER.info("Running the sql code")
    state['state'] = 'sql_code_runner'
    code_str = state['results']['write_sql_code']

    LOGGER.warning("SQL code runner is not implemented yet. Need to register the SQL engine first.")
    state['results']["sql_code_output"] = "SQL code runner is not implemented yet. Need to register the SQL engine first."
    return state


########################################################################################
# Agent Function: summary_agent
# Description: This function is used to summarize the code
########################################################################################
def summary_agent(state: AgentState):
    LOGGER.info("Summarizing the code")
    state['state'] = 'sumary'
    
    text = f"User: {state['results']['user']}\nPython Code: {state['results']['write_python_code']}\nSQL Code: {state['results']['write_sql_code']}"
    result = Engines.get_summary_engine().run(
        [
            f"You are an AI assistant. Your task is to summarize the given text.",
            f"Given text is: {text}"
            f"Now, summarize the given text. Use the tool 'Summarization' for this. Strictly follow the the arguments and return the result."
        ]
    )[0]
    state['results']["summary"] = result['summary']
    return state


########################################################################################
# Agent Function: error_agent
# Description: This function is used to handle the error
########################################################################################
def error_agent(state: AgentState):
    LOGGER.error("Error in the user query")
    state['state'] = 'error'
    return state


########################################################################################
# Agent Function: end_agent
# Description: This function is used
########################################################################################
def end_agent(state: AgentState):
    LOGGER.info("Ending the agent-assist")
    state['state'] = 'end'
    return state