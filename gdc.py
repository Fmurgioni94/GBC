from cat.mad_hatter.decorators import hook
import json
# import re
from cat.log import log


# Constants for JSON templates
JSON_TEMPLATE = '''
{
    "tasks": [
        {
            "name_of_the_task": "task name",
            "estimated_duration": "minutes as integer",
            "description": "detailed task description"
        }
    ]
}
'''

json_data = ""
dictionary_of_tasks = []

def clearing_input(output):
    """
    Process and clean JSON output from LLM response.
    """
    try:
        json_data = output
        clear_json = json_data.replace("```json", "").replace("```", "").strip()
        tasks_list = json.loads(clear_json)
        tasks_dict = {f"tasks-{i}": task for i, task in enumerate(tasks_list)}
        return tasks_dict
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse JSON: {e}")
        return {}
    except Exception as e:
        log.error(f"Unexpected error in clearing_input: {e}")
        return {}
    
@hook 
def before_cat_reads_message(user_message_json, cat):
    try:
        # Parse the JSON string into a Python dict
        message_data = json.loads(user_message_json.text)
        message = message_data["message"]
        preferences = message_data["preferences"]
        
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
            "Unbreakable": ["Write a poetry", "What is the weather in London", "What is your", "tell me a joke", "what is the capital of France", "I want to fly to the moon"]
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
                4. Ensure the narrative is detailed, coherent, and tailored to the user’s specific context.

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
            
            user_message_json["text"] = goal_breaking_prompt
            cat.working_memory.hacked = True
        else:
            cat.working_memory.hacked = True
            user_message_json["text"] = "To this message answer: I cannot assist you with this request, the provide input is not suitable for task breaking. Try again with a different input."
            
        return user_message_json
        
    except json.JSONDecodeError as e:
        log.error(f"JSON parsing error: {e}")
        user_message_json["text"] = f"I'm sorry, there was an error processing your request. the error is: {e}"
        cat.working_memory.hacked = True
        return user_message_json
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        user_message_json["text"] = f"An unexpected error occurred. the error is: {e}"
        cat.working_memory.hacked = True
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
