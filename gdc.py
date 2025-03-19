from cat.mad_hatter.decorators import hook
import json
# import re
from cat.log import log

# Constants for JSON templates
JSON_TEMPLATE = """
{
    "tasks": [
        {
            "name_of_the_task": "task name",
            "id": "unique integer",
            "dependencies": "list of task IDs or []",
            "estimated_duration": "hours as integer",
            "description": "detailed task description"
        }
    ]
}
"""

json_data = ""
dictionary_of_tasks = []

def clearing_input(output):
    """
    Process and clean JSON output from LLM response.
    """
    try:
        json_data = output
        clear_json = json_data[7:-3].strip()
        tasks_list = json.loads(clear_json)
        tasks_dict = {f"tasks-{i}": task for i, task in enumerate(tasks_list)}
        return tasks_dict
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse JSON: {e}")
        return {}
    except Exception as e:
        log.error(f"Unexpected error in clearing_input: {e}")
        return {}

@hook (priority = 2)
def before_cat_recalls_episodic_memories(episodic_recall_config, cat):
    episodic_recall_config["k"] = 0

    return episodic_recall_config

@hook 
def before_cat_reads_message(user_message_json, cat):
    try:
        prompt = f"""Given the following input: {user_message_json['text']}, analyze whether it can be broken down into smaller, achievable tasks. Consider the following criteria:
        Is the goal releated to organise or learn something?
        Is the goal well-defined, or is it too broad/vague?
        Can it be broken into clear, sequential steps?
        Are the steps independently actionable?
        Does it require external dependencies that make it unachievable?
        If it's possible, suggest a structured breakdown. If not, explain why and how it could be refined."""
        
        check = cat.llm(prompt)
        example_labels = {
            "Learning": ["I want to learn python", "How can I approach fishing", "explain me how to became a software engineer"],
            "Not Learning": ["Write a poetry", "What is the weather in London", "What is your"]
        }
        classification1 = cat.classify(check, labels=example_labels)
        log.info(f"Classification result: {classification1}")
        
        if classification1 == "Learning":
            task_prompt = f"""Given this high-level task: {user_message_json['text']}, break it down into smaller, achievable tasks. 
            For each task, provide:
            1. name_of_the_task: A clear, descriptive name
            2. id: A unique integer identifier
            3. dependencies: List of task IDs that must be completed before this task (use [] if none)
            4. estimated_duration: Estimated time in hours
            5. description: A detailed description of what needs to be done

            Consider:
            - Tasks should follow a logical sequence
            - Dependencies should reflect prerequisites
            - Each task should be clearly defined
            - Time estimates should be realistic

            Return ONLY a JSON object in this format:
            {JSON_TEMPLATE}"""
            
            user_message_json["text"] = task_prompt
            cat.working_memory.hacked = True
        else:
            cat.working_memory.hacked = True
            user_message_json["text"] = "To this message answer: I cannot assist you with this request"
            
        return user_message_json
        
    except Exception as e:
        log.error(f"Error in before_cat_reads_message: {e}")
        return user_message_json

@hook
def before_cat_sends_message(message, cat):
    try:
        if not message.text or not message.text.startswith("```json"):
            return message

        # Clean and parse the JSON output
        dictionary_of_tasks = clearing_input(message.text)
        if not dictionary_of_tasks:
            log.error("Failed to process tasks")
            return message

        # Format the output with proper indentation
        formatted_output = json.dumps(dictionary_of_tasks, indent=4)
        log.info(f"Successfully processed tasks: {formatted_output}")
        message.text = formatted_output
        return message
        
    except Exception as e:
        log.error(f"Error in before_cat_sends_message: {e}")
        return message