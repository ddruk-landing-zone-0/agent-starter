import os
import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account
import requests
from ..common.pydantic_service import pydantic_schema_to_tool_format
from ..common.logger import LOGGER
import time


####################################################################################################
# The following code is used to create a GeminiModel class that can be used to interact with the Gemini API.
# Refer: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
# For Rate Limiting: https://ai.google.dev/gemini-api/docs/rate-limits
####################################################################################################

def make_request_with_retries(max_retries=5, wait_time=30, *args, **kwargs):
    retries = 0
    while retries < max_retries:
        response = requests.post(*args, **kwargs)
        
        if response.status_code == 429:
            retries += 1
            LOGGER.warning(f"Rate limit hit. Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
            time.sleep(wait_time)
        else:
            return response  # Return response if successful or any other error
    
    LOGGER.critical(f"Failed after {max_retries} retries due to rate limiting.")
    raise Exception(f"Failed after {max_retries} retries due to rate limiting.")


class GeminiModel:
    def __init__(self, model_name, temperature=0.7, max_output_tokens=1024, max_retries=5, wait_time=30, deployed_gcp=False):
        """
        Initialize the GeminiModel with model configuration.

        :param model_name: The model identifier (e.g., "gemini-1.0-pro-002")
        :param temperature: Sampling temperature for response diversity.
        :param max_output_tokens: Max number of tokens in the response.
        """
        try:
            try:
                if deployed_gcp:
                    self._access_token = self._get_access_token_gcp()
                    self._project_id = self._get_project_id_gcp()
                    self._project_location = self._get_project_location_gcp()
                else:
                    self._access_token = self._get_access_token()
                    self._project_id = self._get_project_id()
                    self._project_location = self._get_project_location()
            except Exception as e:
                LOGGER.error(f"Failed to authenticate: {str(e)}")
                raise RuntimeError(f"Failed to authenticate: {str(e)}")

            self._model_name = model_name
            self.temperature = temperature
            self.max_output_tokens = max_output_tokens
            self.max_retries = max_retries
            self.wait_time = wait_time
            self.headers = {
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/json"
            }

            LOGGER.debug(f"Initialized GeminiModel with model {model_name} , project {self._project_id}, location {self._project_location}")
        except Exception as e:
            LOGGER.error(f"Failed to initialize GeminiModel: {str(e)}")
            raise RuntimeError(f"Failed to initialize GeminiModel: {str(e)}")

    ####################################################################################################
    # The following authentication methods are used for service account authentication from outside a GCP Environment using a service account key.
    def _get_access_token(self):
        """Retrieve an access token using service account credentials."""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            credentials.refresh(google.auth.transport.requests.Request())
            return credentials.token
        except Exception as e:
            raise RuntimeError(f"Error obtaining access token: {str(e)}")

    def _get_project_id(self):
        """Retrieve the project ID from environment variables or config."""
        return os.environ["GOOGLE_CLOUD_PROJECT"]

    def _get_project_location(self):
        """Retrieve the project location from environment variables or config."""
        return os.environ["GOOGLE_CLOUD_LOCATION"]
    

    ####################################################################################################
    # The following authentication methods are used for GCP metadata server authentication from inside a deployed GCP Environment.
    def _get_access_token_gcp(self):
        """Fetches an access token for authentication with Vertex AI."""
        METADATA_URL = "http://metadata.google.internal/computeMetadata/v1"
        response = requests.get(f"{METADATA_URL}/instance/service-accounts/default/token", headers={"Metadata-Flavor": "Google"})
        return response.json()["access_token"]
    
    def _get_metadata_gcp(self, path):
        """Fetches metadata from GCP metadata server."""
        METADATA_URL = "http://metadata.google.internal/computeMetadata/v1"
        response = requests.get(f"{METADATA_URL}/{path}", headers={"Metadata-Flavor": "Google"})
        return response.text

    def _get_project_id_gcp(self):
        return self.get_metadata("project/project-id")

    def _get_project_location_gcp(self):
        full_zone = self.get_metadata("instance/zone")  # e.g., projects/12345/zones/us-central1-a
        return full_zone.split("/")[-1].rsplit("-", 1)[0]  # Extracts 'us-central1' from 'us-central1-a'

    

    def _validate_args(self, arg, type:str):
        if type == "content_role_list":
            if not isinstance(arg, list):
                raise ValueError("Content role list must be a list")
            for item in arg:
                if "role" not in item or "content" not in item:
                    raise ValueError("Content role list must have 'role' and 'content' keys")
                if item["role"] not in ["user", "model"]:
                    raise ValueError("Content role must be 'user' or 'model'")
        elif type == "system_instructions":
            if not isinstance(arg, str):
                raise ValueError("System instructions must be a string")
        elif type == "tools":
            if not isinstance(arg, list):
                raise ValueError("Tools must be a list")
            for tool in arg:
                if "name" not in tool or "description" not in tool or "parameters" not in tool:
                    raise ValueError("Tool must have 'name', 'description', and 'parameters' keys")
                if not isinstance(tool["name"], str) or not isinstance(tool["description"], str) or not isinstance(tool["parameters"], dict):
                    raise ValueError("Tool keys must be strings or dict")
        else:
            LOGGER.error(f"Invalid argument type: {type}")
            raise ValueError(f"Invalid argument type: {type}")

    def _create_payload_for_generate(self, content_role_list, system_instructions=None):
        """
        Create a structured payload for content generation.

        :param content_role_list: List of dicts with role (system/user/assistant) and content.
        :param system_instructions: Optional system instructions.
        :return: JSON payload for request.
        """
        try:
            # Validate content
            self._validate_args(content_role_list, "content_role_list")
            if system_instructions:
                self._validate_args(system_instructions, "system_instructions")

            payload = {
                "contents": [
                    {
                        "role": item["role"],
                        "parts": [{"text": item["content"]}]
                    }
                    for item in content_role_list
                ],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_output_tokens
                }
            }
            if system_instructions:
                payload["systemInstruction"] = {
                    "role": "system",
                    "parts": [{"text": system_instructions}]
                }
            return payload
        except Exception as e:
            LOGGER.error(f"Error constructing payload: {str(e)}")
            raise ValueError(f"Error constructing payload: {str(e)}")

    def _create_payload_for_generate_funccall(self, content_role_list, tools, system_instructions=None):
        """
        Create a structured payload for function calling.

        :param content_role_list: List of dicts with role (system/user/assistant) and content.
        :param tools: List of tool function objects, each following a structured schema.
        :return: JSON payload for request.
        """
        try:
            # Validate content
            self._validate_args(content_role_list, "content_role_list")
            self._validate_args(tools, "tools")
            if system_instructions:
                self._validate_args(system_instructions, "system_instructions")
            

            payload = {
                "contents": [
                    {
                        "role": item["role"],
                        "parts": [{"text": item["content"]}]
                    }
                    for item in content_role_list
                ],
                "tools": [{
                    "functionDeclarations": [
                        {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": tool["parameters"]
                        }
                    for tool in tools]
                }]
            }

            if system_instructions:
                payload["systemInstruction"] = {
                    "role": "system",
                    "parts": [{"text": system_instructions}]
                }

            return payload
        except Exception as e:
            LOGGER.error(f"Error constructing function call payload: {str(e)}")
            raise ValueError(f"Error constructing function call payload: {str(e)}")

    def generate_content(self, content_role_list, system_instructions=None,simplify_output=False):
        """
        Send a request for content generation.

        :param content_role_list: List of dicts with role and content.
        :param system_instructions: Optional system-level instructions.
        :return: Generated content response.
        """
        try:
            url = f"https://{self._project_location}-aiplatform.googleapis.com/v1/projects/{self._project_id}/locations/{self._project_location}/publishers/google/models/{self._model_name}:generateContent"
            payload = self._create_payload_for_generate(content_role_list, system_instructions)
            response = make_request_with_retries(self.max_retries, self.wait_time, url, headers=self.headers, json=payload)
            response.raise_for_status()
            response = response.json()
            if simplify_output:
                response = response["candidates"][0]['content']['parts'][0]['text']
            return response
        except requests.exceptions.RequestException as e:
            LOGGER.error(f"Error generating content: {str(e)}")
            raise RuntimeError(f"Error generating content: {str(e)}")

    def generate_funccall_content(self, content_role_list, tools, system_instructions=None, simplify_output=False):
        """
        Send a request for function calling.

        :param content_role_list: List of dicts with role and content.
        :param tools: List of tool function objects.
        :return: Function call response.
        """
        try:
            url = f"https://{self._project_location}-aiplatform.googleapis.com/v1/projects/{self._project_id}/locations/{self._project_location}/publishers/google/models/{self._model_name}:generateContent"
            payload = self._create_payload_for_generate_funccall(content_role_list, tools, system_instructions)
            response = make_request_with_retries(self.max_retries, self.wait_time, url, headers=self.headers, json=payload)
            response.raise_for_status()
            response = response.json()
            if simplify_output:
                response = [r['functionCall'] for r in response["candidates"][0]['content']['parts']]
            
            return response
        except requests.exceptions.RequestException as e:
            LOGGER.error(f"Error generating function call content: {str(e)}")
            raise RuntimeError(f"Error generating function call content: {str(e)}")




####################################################################################################
# The following code is used to create a GeminiJsonEngine class that can be used to interact with the Gemini API.
# It is a wrapper around the GeminiModel class that simplifies strutured output.
####################################################################################################

class GeminiJsonEngine():
    def __init__(self, model_name, basemodel, temperature, max_output_tokens, systemInstructions, max_retries=5, wait_time=30, deployed_gcp=False):
        self.model = GeminiModel(model_name=model_name, temperature=temperature, max_output_tokens=max_output_tokens, max_retries=max_retries, wait_time=wait_time, deployed_gcp=deployed_gcp)
        self.content_roles = []
        if systemInstructions:
            self.content_roles.append({"role": "user", "content": systemInstructions})
        self.schema = pydantic_schema_to_tool_format(basemodel)
    
    def run(self, user_query):
        """
        Input: user_query: List[str] or str
        Output: response: List[Dict]
        """
        if isinstance(user_query, str):
            _content_roles = self.content_roles + [{"role": "user", "content": user_query}]
        elif isinstance(user_query, list):
            _content_roles = self.content_roles + [{"role": "user", "content": item} for item in user_query]
        else:
            raise ValueError("Input must be a string or list")
        response = self.model.generate_funccall_content(_content_roles, tools=[self.schema], simplify_output=True)
        return [r['args'] for r in response]
    


####################################################################################################
# The following code is used to create a GeminiSimpleChatEngine class that can be used to interact with the Gemini API.
# It is a wrapper around the GeminiModel class that simplifies simple chat interactions.
####################################################################################################

class GeminiSimpleChatEngine:
    def __init__(self, model_name, temperature, max_output_tokens, systemInstructions, max_retries=5, wait_time=30, deployed_gcp=False):
        self.model = GeminiModel(model_name=model_name, temperature=temperature, max_output_tokens=max_output_tokens, max_retries=max_retries, wait_time=wait_time, deployed_gcp=deployed_gcp)
        self.content_roles = []
        if systemInstructions:
            self.content_roles.append({"role": "user", "content": systemInstructions})
    
    def run(self, user_query):
        """
        Input: user_query: List[str] or str
        Output: response: str
        """
        if isinstance(user_query, str):
            _content_roles = self.content_roles + [{"role": "user", "content": user_query}]
        elif isinstance(user_query, list):
            _content_roles = self.content_roles + [{"role": "user", "content": item} for item in user_query]
        else:
            raise ValueError("Input must be a string or list")
        
        response = self.model.generate_content(_content_roles, simplify_output=True)
        return response