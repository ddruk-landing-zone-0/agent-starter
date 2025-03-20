
from ..llms.gemini import GeminiJsonEngine
from .agent_baseclass import QueryClassfication, PythonCode, SqlCode, Summarization
from functools import lru_cache


class Engines:
    @staticmethod
    @lru_cache(maxsize=None)  # Unlimited cache
    def get_check_user_qry_engine():
        check_user_qry_engine = GeminiJsonEngine(
                                    model_name="gemini-2.0-flash-001",
                                    basemodel=QueryClassfication,
                                    temperature=0.5,
                                    max_output_tokens=256,
                                    systemInstructions=None,
                                    max_retries=5,
                                    wait_time=30,
                                    deployed_gcp=False
                                    )
        return check_user_qry_engine
    
    @staticmethod
    @lru_cache(maxsize=None)  # Unlimited cache
    def get_generate_python_code_engine():
        generate_python_code_engine = GeminiJsonEngine(
                                    model_name="gemini-2.0-flash-001",
                                    basemodel=PythonCode,
                                    temperature=0.5,
                                    max_output_tokens=256,
                                    systemInstructions=None,
                                    max_retries=5,
                                    wait_time=30,
                                    deployed_gcp=False
                                    )
        return generate_python_code_engine
    
    @staticmethod
    @lru_cache(maxsize=None)  # Unlimited cache
    def get_generate_sql_code_engine():
        generate_sql_code_engine = GeminiJsonEngine(
                                    model_name="gemini-2.0-flash-001",
                                    basemodel=SqlCode,
                                    temperature=0.5,
                                    max_output_tokens=256,
                                    systemInstructions=None,
                                    max_retries=5,
                                    wait_time=30,
                                    deployed_gcp=False
                                    )
        return generate_sql_code_engine
    
    @staticmethod
    @lru_cache(maxsize=None)  # Unlimited cache
    def get_summary_engine():
        summary_engine = GeminiJsonEngine(
                                    model_name="gemini-2.0-flash-001",
                                    basemodel=Summarization,
                                    temperature=0.5,
                                    max_output_tokens=256,
                                    systemInstructions=None,
                                    max_retries=5,
                                    wait_time=30,
                                    deployed_gcp=False
                                    )
        return summary_engine