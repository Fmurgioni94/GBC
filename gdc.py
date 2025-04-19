from cat.mad_hatter.decorators import hook, tool
import json
import traceback
from cat.log import log
from cat.looking_glass.output_parser import LLMAction


# Constants for JSON templates
JSON_TEMPLATE = """{"tasks": {"id": "unique identifier for the task", "name_of_the_task": "task name","estimated_duration": "minutes as integer", "description": "detailed task description", "dependencies": must be an integer not a string with a integer inside just a integer use the taks id of the task that must be completed before this one or "" if there are no dependencies eg: 1, 2, [1, 2], ""} }"""

INPUT_CHECK_TEMPLATE = """
'{"message": "I want to learning python", "preferences": {"learningStyle": "visual", "preferredComplexity": "medium", "motivation": "high", "priorExperience": "beginner", "additionalNotes": ""} }'
"""


json_data = ""
dictionary_of_tasks = []

def clearing_input(output):
    """
    Process and clean JSON output from LLM response.
    Safely handles both string inputs and Pydantic models.
    """
    try:
        # If it's already a dict, return it directly
        if isinstance(output, dict):
            return output
            
        # If it's a Pydantic model, convert to dict safely
        if hasattr(output, 'model_dump'):
            try:
                return output.model_dump()
            except Exception as e:
                log.error(f"Error converting Pydantic model to dict: {e}")
                return {}
                
        # If it's a string, try to parse as JSON
        if isinstance(output, str):
            try:
                # Clean the string first
                clean_str = output.replace("```json", "").replace("```", "").strip()
                # Validate it's not empty after cleaning
                if not clean_str:
                    log.error("Empty string after cleaning")
                    return {}
                # Parse JSON with strict validation
                return json.loads(clean_str)
            except json.JSONDecodeError as e:
                log.error(f"Invalid JSON format: {e}")
                log.error(f"Problematic input: {clean_str}")
                return {}
                
        # If we get here, we don't know how to handle the input
        log.error(f"Unsupported input type: {type(output)}")
        return {}
        
    except Exception as e:
        log.error(f"Unexpected error in clearing_input: {e}")
        return {}
    
def process_json_message(message_text):
    """
    Process and format JSON messages.
    Safely handles both string inputs and dictionary outputs from clearing_input.
    """
    try:
        # Handle empty or invalid input
        if not message_text:
            log.error("Empty message text received")
            return "{}"
            
        # If it's already a dictionary, format it directly
        if isinstance(message_text, dict):
            try:
                return json.dumps(message_text, indent=4)
            except Exception as e:
                log.error(f"Error formatting dictionary: {e}")
                return "{}"
                
        # If it's a string, process it through clearing_input
        if isinstance(message_text, str):
            # Only process if it's marked as JSON
            if not message_text.startswith("```json"):
                return message_text
                
            dictionary_of_tasks = clearing_input(message_text)
            if not dictionary_of_tasks:
                log.error("Failed to process tasks")
                return "{}"

            try:
                # Ensure the output is a valid JSON string
                formatted_output = json.dumps(dictionary_of_tasks, indent=4)
                log.info(f"Successfully processed tasks: {formatted_output}")
                return formatted_output
            except Exception as e:
                log.error(f"Error formatting JSON: {e}")
                return "{}"
                
        # If we get here, we don't know how to handle the input
        log.error(f"Unsupported input type: {type(message_text)}")
        return "{}"
        
    except Exception as e:
        log.error(f"Unexpected error in process_json_message: {e}")
        return "{}"
    

@tool(return_direct=True, examples=["I want to learning python", "I want to becoming a software engineer", "I want to learn to code", "I want to learn to code in python"])
def break_down_task(user_message_json: str, cat):
    """
    Use this tool when the user has to break down a goal into smaller, achievable tasks. As input use the entire message recived.
    """
    input_prompt = f"""
    Clean the following message from any extra characters and use the information in it to fill the template below then return it as a json object.
    message: {user_message_json}
    template: {INPUT_CHECK_TEMPLATE}
    """
    clean_input = cat.llm(input_prompt)
    log.info(f"Received user input: {clean_input}")
    try:
        # Clean and validate the input JSON
        if not clean_input:
            return "Error: Empty input received"  # Return empty JSON object instead of error message
            
        # Remove any potential escape characters and whitespace
        cleaned_json = clean_input.replace("```json", "").replace("```", "").strip()
        
        # Remove outer quotes if present
        if cleaned_json.startswith('"') and cleaned_json.endswith('"'):
            cleaned_json = cleaned_json[1:-1]
            
        # Validate JSON structure
        try:
            message_data = json.loads(cleaned_json)
            log.info(f"Parsed message data: {message_data}")
        except json.JSONDecodeError as e:
            log.error(f"JSON parsing error: {e}")
            log.error(f"Problematic JSON: {cleaned_json}")
            return "{}"  # Return empty JSON object instead of error message
            
        # Validate required fields
        if not isinstance(message_data, dict):
            return "Error: Invalid input format"
            
        if "message" not in message_data:
            return "Error: Missing message in input"
            
        if "preferences" not in message_data:
            return "Error: Missing preferences in input"
            
        message = message_data["message"]
        preferences = message_data["preferences"]
        
        cat.send_notification("break_down_task tool called")

        try:
            prompt = f"""
                You are tasked with analyzing the following input to determine if it can be broken down into smaller, achievable tasks. Evaluate the input based on the criteria below:

                Input: {message}

                Evaluation Criteria:
                1. Is the goal related to organizing or learning something?
                2. Can the goal be broken into clear, sequential steps?
                3. Are the identified steps independently actionable?
                4. Does the goal require external dependencies that would make it unachievable?

                Provide a detailed analysis of the input, addressing each criterion and explaining your reasoning.
            """
            
            check = cat.llm(prompt)
            example_labels = {
                "Breakable": ["I want to learn python", "How can I approach fishing", "explain me how to became a software engineer", "How can i learn to code"],
                "Unbreakable": ["Write a poetry", "What is the weather in London", "What is your name", "tell me a joke", "what is the capital of France", "I want to fly to the moon"]
            }
            classification1 = cat.classify(check, labels=example_labels)
            log.info(f"Classification result: {classification1}")
            
            if classification1 == "Breakable":
                generate_task_description_prompt = f"""
                    You are tasked with generating a comprehensive and structured description of a user's goal. Use the information provided below to create a detailed narrative that reflects all aspects of the user's needs and preferences.

                    User Information:
                    - **User Goal:** {message}
                    - **Preferred Learning Style:** {preferences["learningStyle"]}
                    - **Preferred Complexity Level:** {preferences["preferredComplexity"]}
                    - **Motivational Level:** {preferences["motivation"]}
                    - **Prior Experience:** {preferences["priorExperience"]}
                    - **Additional Notes:** {preferences["additionalNotes"]}

                    Instructions:
                    1. Analyze each piece of information and explain how it influences the overall goal.
                    2. Organize your description into clear sections (e.g., Overview, Detailed Explanation, and Implications).
                    3. Emphasize any interconnections between the user's learning style, preferred complexity, motivational level, and prior experience.
                    4. Ensure the narrative is detailed, coherent, and tailored to the user's specific context.

                    Generate the final description of the user goal.
                """

                refined_goal_description = cat.llm(generate_task_description_prompt)
                log.info(f"Refined goal description: {refined_goal_description}")

                goal_breaking_prompt = f"""
                    You are tasked with generating a series of smaller, achievable subtasks to help a user organize their learning process based on the high-level task provided below.

                    High-Level Task:
                    - {refined_goal_description}

                    Instructions:
                    1. Decompose the high-level task into a logical sequence of subtasks.
                    2. For each subtask, provide:
                    - **name_of_the_task:** A clear, descriptive title for the subtask.
                    - **description:** A detailed explanation of what needs to be done, including practical examples and tips.
                    3. Ensure that:
                    - The subtasks follow a logical order.
                    - Each subtask is clearly defined.
                    - Time estimates are realistic for a student with {preferences["priorExperience"]} prior experience in the topic.
                    - If a task is too time-consuming, split it into smaller, manageable subtasks.

                    Output Requirements:
                    - Return ONLY a JSON object that exactly follows this format:
                    {JSON_TEMPLATE}
                """

                tasks_breakdown = cat.llm(goal_breaking_prompt)
                cat.send_notification("Tasks breakdown in progress")
                processed_message = process_json_message(tasks_breakdown)
                cat.send_notification("Tasks breakdown completed")
                # Ensure the output is valid JSON
                try:
                    json.loads(processed_message)
                    return processed_message
                except json.JSONDecodeError:
                    log.error("Invalid JSON returned from process_json_message")
                    return "{}"
                
            else:
                cat.send_notification("The task is not breakable")
                log.info("The task is not breakable")
                return "{}"
                
        except Exception as e:
            log.error(f"Error in task breakdown: {e}")
            return "{}"
            
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        return f"An unexpected error occurred. the error is: {e}"
    
@hook(priority=1)
def before_cat_sends_message(message: dict, cat) -> dict:
    """
    Hook to process messages before they are sent to the user.
    Only modifies the message if no tools were used.    
    """
    try:
        # Check model interactions to see what actions were taken
        model_interactions = cat.working_memory.model_interactions
        if not model_interactions:
            return message

        # Look for ProceduresAgent interaction to see what action was chosen
        for interaction in model_interactions:
            if interaction.source == 'ProceduresAgent':
                try:
                    # Parse the LLM's response to get the chosen action
                    response = interaction.reply
                    clear_json = response.replace("```json", "").replace("```", "").strip()
                    action_data = json.loads(clear_json)
                    
                    # If action is "no_action", show our message
                    if action_data.get("action") == "no_action":
                        message["text"] = "I'm sorry, but I couldn't process your request using any of my tools. Please try rephrasing your question."
                    
                    break
                except:
                    continue

        return message

    except Exception as e:
        log.error(f"Error in before_cat_sends_message: {e}")
        return message
