import langgraph
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver

from .agent_state import AgentState
from .agent_functions import start_agent, check_user_qry, write_python_code_agent, python_code_runner_agent, write_sql_code_agent, sql_code_runner_agent, summary_agent, error_agent, end_agent


class MyAgent:
    def __init__(self, thread_id=None):
        self.config = None
        self.app = None
        self.build(thread_id)

    def build(self, thread_id):
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node('start', start_agent)
        workflow.add_node('write_python_code', write_python_code_agent)
        workflow.add_node('write_sql_code', write_sql_code_agent)
        workflow.add_node('python_code_runner', python_code_runner_agent)
        workflow.add_node('sql_code_runner', sql_code_runner_agent)
        workflow.add_node('summary', summary_agent)
        workflow.add_node('error', error_agent)
        workflow.add_node('end', end_agent)

        # Edges
        workflow.add_conditional_edges(
            'start', 
            check_user_qry,
            {
                "python": "write_python_code",
                "sql": "write_sql_code",
                "other": "error"
            }
        )
        workflow.add_edge('write_python_code', 'python_code_runner')
        workflow.add_edge('python_code_runner', 'summary')
        workflow.add_edge('write_sql_code', 'sql_code_runner')
        workflow.add_edge('sql_code_runner', 'summary')
        workflow.add_edge('error', 'end')
        workflow.add_edge('summary', 'end')
        

        # Compile
        workflow.set_entry_point('start')
        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)
        self.config = {"configurable":{"thread_id":str(thread_id)}}

    def get_recent_state_snap(self):
        snap = self.app.get_state(self.config).values.copy()
        return snap
    
    def get_graph(self):
        graph = self.app.get_graph(xray=True)
        return graph
    
    def continue_flow(self, state):
        self.app.invoke(state, config=self.config)
        return self.get_recent_state_snap()