# Goal Breakdown Component (GBC)

A Cheshire Cat AI plugin designed to help users break down complex learning goals into manageable, actionable tasks.

## Overview

The GBC plugin analyzes user input to determine if a goal can be broken down into smaller, achievable steps. It's particularly useful for learning and organizational goals, providing structured task breakdowns with detailed descriptions and time estimates.

## Features

### Core Functionality
- **Goal Analysis**: Determines if a goal can be broken down into smaller tasks (Implemented in `gdc.py:break_down_task()`)
- **Task Generation**: Creates detailed, actionable subtasks (Implemented in `gdc.py:process_json_message()`)
- **Preference Integration**: Considers user preferences and learning styles (Handled in task generation prompts)
- **Time Estimation**: Provides realistic time estimates for each task (Part of LLM task breakdown)

### User Preferences Support
All preferences are processed in `gdc.py:break_down_task()`:
- Learning Style
- Preferred Complexity Level
- Motivation Level
- Prior Experience
- Additional Notes

## Technical Details

### Expected Input Format
```json
{
    "message": "I want to learn Python",
    "preferences": {
        "learningStyle": "visual",
        "preferredComplexity": "medium",
        "motivation": "high",
        "priorExperience": "beginner",
        "additionalNotes": ""
    }
}
```

### Output JSON Structure
The following format is returned to the frontend after task breakdown:
```json
{
    "tasks": {
        "id": "unique task identifier",
        "name_of_the_task": "task name",
        "estimated_duration": "minutes as integer",
        "description": "detailed task description",
        "dependencies": [1, 2]  // IDs of prerequisite tasks
    }
}
```

### LLM Usage
The plugin leverages LLMs in several key steps (all implemented in `gdc.py:break_down_task()`):

1. **Goal Analysis**
   ```python
   prompt = f"""
       You are tasked with analyzing the following input: {message}
       Evaluation Criteria:
       1. Is the goal related to organizing or learning something?
       2. Can the goal be broken into clear, sequential steps?
       3. Are the identified steps independently actionable?
       4. Does the goal require external dependencies?
   """
   check = cat.llm(prompt)
   ```

2. **Task Classification**
   ```python
   example_labels = {
       "Breakable": ["I want to learn python", "How can I approach fishing"],
       "Unbreakable": ["What is the weather", "tell me a joke"]
   }
   classification1 = cat.classify(check, labels=example_labels)
   ```

3. **Goal Description Generation**
   ```python
   generate_task_description_prompt = f"""
       User Information:
       - **User Goal:** {message}
       - **Preferred Learning Style:** {preferences["learningStyle"]}
       - **Preferred Complexity Level:** {preferences["preferredComplexity"]}
       - **Motivational Level:** {preferences["motivation"]}
       - **Prior Experience:** {preferences["priorExperience"]}
       - **Additional Notes:** {preferences["additionalNotes"]}
   """
   refined_goal_description = cat.llm(generate_task_description_prompt)
   ```

4. **Task Breakdown**
   ```python
   goal_breaking_prompt = f"""
       You are tasked with generating a series of smaller, achievable subtasks...
       High-Level Task: {refined_goal_description}
       Instructions:
       1. Decompose the high-level task into a logical sequence of subtasks...
   """
   tasks_breakdown = cat.llm(goal_breaking_prompt)
   ```

**Prompt Engineering:**
- Each prompt is carefully structured to guide the LLM toward specific outputs
- Examples are provided for classification to improve accuracy
- User preferences are incorporated into prompts to personalize task generation
- Error handling is built into each LLM interaction

### Caching and Performance
Currently, the plugin processes each goal from scratch. However, caching is planned for future versions:

**Planned Caching Features:**
- Cache similar goal breakdowns to reduce LLM calls
- Store task templates for common learning paths
- Implement similarity matching for existing breakdowns
- Add configuration options for cache duration and size

**Current Performance Considerations:**
- Each request requires multiple LLM calls
- Processing time depends on goal complexity
- No persistent storage of previous breakdowns

### Task Classification
A goal is considered "unbreakable" if it meets any of these criteria:
- Not learning-oriented (e.g., "What's the weather?")
- Single-step action (e.g., "Tell me a joke")
- Non-actionable query (e.g., "What is your name?")
- Requires external dependencies beyond user control

Implementation: `gdc.py:break_down_task()` uses example-based classification

### Error Handling
The plugin handles various error types (Implemented in `gdc.py`):
- **JSON Parsing Errors**: Returns `{}` with error logged when input JSON is malformed
- **Missing Fields**: Returns specific error message for missing required fields
- **Invalid Input Format**: Returns error message for non-JSON or incorrectly structured input
- **Processing Failures**: Logs error and returns fallback message to user

### Constraints and Limitations
- Maximum tasks per breakdown: No hard limit, controlled by LLM
- Task Dependencies: Handled through `dependencies` field in output
- Time Estimates: Provided in minutes, no overlap checking
- Task Ordering: Sequential based on dependencies

### Visual Flow

## Installation

1. Place the `GBC` folder in your Cheshire Cat plugins directory:
   ```
   core/cat/plugins/GBC/
   ```

2. Ensure the following files are present:
   - `gdc.py`: Main plugin code
   - `__init__.py`: Plugin initialization
   - `plugin.json`: Plugin configuration
   - `requirements.txt`: Dependencies

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Restart the Cheshire Cat service

## Usage

### Basic Usage
1. Send a message containing a learning or organizational goal
2. The plugin will analyze if the goal can be broken down
3. If possible, it will return a structured breakdown of tasks
4. If not suitable for breakdown, it will provide an explanation

### Example Input
```
"I want to learn Python programming"
```

### Example Output
```json
{
    "tasks": {
        "name_of_the_task": "Python Basics",
        "estimated_duration": "120",
        "description": "Learn basic Python syntax, variables, and data types"
    }
}
```

## Configuration

The plugin can be configured through `plugin.json`:

```json
{
    "name": "Goal Breakdown Component",
    "description": "Breaks down learning goals into manageable tasks",
    "version": "1.0.0",
    "author": "Your Name",
    "entry_point": "gdc"
}
```

## Customization

The plugin can be extended through `plugin.json`:
```json
{
    "name": "Goal Breakdown Component",
    "description": "Breaks down learning goals into manageable tasks",
    "version": "1.0.0",
    "author": "Your Name",
    "entry_point": "gdc",
    "settings": {
        "max_tasks": 10,        // Optional: limit max tasks
        "min_duration": 15,     // Optional: minimum task duration
        "enable_dependencies": true  // Optional: enable/disable task dependencies
    }
}
```

## Changelog

### Version 1.0.0
- Initial release
- Basic task breakdown functionality
- User preference integration
- JSON input/output handling
- Error handling implementation

### Version 1.0.1 (Planned)
- Enhanced dependency handling
- Task validation improvements
- Additional customization options

## Error Handling

The plugin includes comprehensive error handling for:
- JSON parsing errors
- Invalid input formats
- Processing failures
- Unexpected errors

## Dependencies

- Python 3.x
- Cheshire Cat AI system
- Required packages (specified in requirements.txt)

## Troubleshooting

Common issues and solutions:
1. **JSON Parsing Errors**: Ensure input follows the correct format
2. **Plugin Not Loading**: Check plugin.json configuration
3. **Dependency Issues**: Verify all requirements are installed

## Support

For issues or questions:
1. Check the Cheshire Cat documentation
2. Review the plugin's error logs
3. Create an issue in the repository

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This plugin is licensed under the same terms as the Cheshire Cat AI project.

## How It Works

The GBC plugin follows a multi-step process to break down learning goals into manageable tasks:

### 1. Input Processing
- User sends a message containing a learning goal
- The message is received in JSON format containing:
  ```json
  {
      "message": "user's goal",
      "preferences": {
          "learningStyle": "...",
          "preferredComplexity": "...",
          "motivation": "...",
          "priorExperience": "...",
          "additionalNotes": "..."
      }
  }
  ```

### 2. Goal Analysis
- The plugin first analyzes if the goal can be broken down using the following criteria:
  1. Is the goal related to organizing or learning something?
  2. Can the goal be broken into clear, sequential steps?
  3. Are the identified steps independently actionable?
  4. Does the goal require external dependencies that would make it unachievable?

### 3. Classification
- The goal is classified into one of two categories:
  - "Breakable": Goals that can be broken down (e.g., "I want to learn Python")
  - "Unbreakable": Goals that cannot be broken down (e.g., "What is the weather?")

### 4. Goal Description Generation
- If the goal is breakable, the plugin:
  1. Analyzes user preferences and learning context
  2. Generates a comprehensive description of the goal
  3. Considers:
     - User's learning style
     - Preferred complexity level
     - Motivation level
     - Prior experience
     - Additional notes

### 5. Task Breakdown
- The plugin then:
  1. Decomposes the goal into logical subtasks
  2. For each subtask, generates:
     - A clear, descriptive title
     - A detailed description
     - Realistic time estimates
  3. Ensures tasks follow a logical sequence
  4. Adjusts complexity based on user's experience level

### 6. Output Generation
- The final output is formatted as JSON:
  ```json
  {
      "tasks": {
          "name_of_the_task": "task name",
          "estimated_duration": "minutes as integer",
          "description": "detailed task description"
      }
  }
  ```

### 7. Error Handling
- The plugin includes comprehensive error handling:
  1. JSON parsing errors
  2. Invalid input formats
  3. Processing failures
  4. Unexpected errors
- Each error is logged and appropriate error messages are returned to the user

### 8. Message Processing
- The `before_cat_sends_message` hook:
  1. Checks if the message contains JSON output
  2. Formats the output for better readability
  3. Ensures proper indentation and structure
  4. Handles any formatting errors

### 9. Tool Integration
- The `break_down_task` tool:
  1. Receives the user's message
  2. Processes it through the breakdown pipeline
  3. Returns the structured task breakdown
  4. Handles any errors during processing

This workflow ensures that:
- Goals are properly analyzed before breakdown
- User preferences are considered
- Tasks are realistic and achievable
- Output is properly formatted and readable
- Errors are handled gracefully

## Implementation Details

Here's how each component is implemented in code:

### Core Components

1. **Plugin Initialization**
```python
from cat.mad_hatter.decorators import hook, tool
import json
import traceback
from cat.log import log
from cat.looking_glass.output_parser import LLMAction

# JSON template for task structure
JSON_TEMPLATE = """{"tasks": {
    "name_of_the_task": "task name",
    "estimated_duration": "minutes as integer",
    "description": "detailed task description"
}}"""
```

2. **Main Tool Implementation**
```python
@tool(return_direct=True, examples=["I want to learning python", "I want to becoming a software engineer"])
def break_down_task(user_message_json: str, cat):
    """Break down a task into smaller, achievable tasks."""
    user_message_json = user_message_json.replace('\\"', '"').strip('"')
    try:
        message_data = json.loads(user_message_json)
        message = message_data["message"]
        preferences = message_data["preferences"]
        # ... rest of implementation
```

3. **JSON Processing Function**
```python
def clearing_input(output):
    """Process and clean JSON output from LLM response."""
    try:
        json_data = output
        clear_json = json_data.replace("```json", "").replace("```", "").strip()
        tasks_list = json.loads(clear_json)
        return tasks_list
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse JSON: {e}")
        return {}
```

4. **Message Processing Function**
```python
def process_json_message(message_text):
    """Process and format JSON messages."""
    try:
        if not message_text or not message_text.startswith("```json"):
            return message_text
        
        dictionary_of_tasks = clearing_input(message_text)
        if not dictionary_of_tasks:
            log.error("Failed to process tasks")
            return message_text

        return json.dumps(dictionary_of_tasks, indent=4)
    except Exception as e:
        log.error(f"Error in process_json_message: {e}")
        return message_text
```

### LLM Integration

1. **Goal Analysis Prompt**
```python
prompt = f"""
    You are tasked with analyzing the following input: {message}
    Evaluation Criteria:
    1. Is the goal related to organizing or learning something?
    2. Can the goal be broken into clear, sequential steps?
    3. Are the identified steps independently actionable?
    4. Does the goal require external dependencies?
"""
check = cat.llm(prompt)
```

2. **Task Classification**
```python
example_labels = {
    "Breakable": ["I want to learn python", "How can I approach fishing"],
    "Unbreakable": ["What is the weather", "tell me a joke"]
}
classification1 = cat.classify(check, labels=example_labels)
```

3. **Task Description Generation**
```python
generate_task_description_prompt = f"""
    User Information:
    - **User Goal:** {message}
    - **Preferred Learning Style:** {preferences["learningStyle"]}
    - **Preferred Complexity Level:** {preferences["preferredComplexity"]}
    - **Motivational Level:** {preferences["motivation"]}
    - **Prior Experience:** {preferences["priorExperience"]}
    - **Additional Notes:** {preferences["additionalNotes"]}
"""
refined_goal_description = cat.llm(generate_task_description_prompt)
```

### Message Processing Hook

```python
@hook(priority=1)
def before_cat_sends_message(message: dict, cat) -> dict:
    """Process messages before they are sent to the user."""
    try:
        model_interactions = cat.working_memory.model_interactions
        if not model_interactions:
            return message

        for interaction in model_interactions:
            if interaction.source == 'ProceduresAgent':
                try:
                    response = interaction.reply
                    clear_json = response.replace("```json", "").replace("```", "").strip()
                    action_data = json.loads(clear_json)
                    
                    if action_data.get("action") == "no_action":
                        message["text"] = "I'm sorry, but I couldn't process your request."
                    break
                except:
                    continue
        return message
    except Exception as e:
        log.error(f"Error in before_cat_sends_message: {e}")
        return message
```

### Error Handling Implementation

```python
try:
    # Process task
    tasks_breakdown = cat.llm(goal_breaking_prompt)
    processed_message = process_json_message(tasks_breakdown)
    return processed_message
except json.JSONDecodeError as e:
    log.error(f"JSON parsing error: {e}")
    return f"I'm sorry, there was an error processing your request: {e}"
except Exception as e:
    log.error(f"Unexpected error: {e}")
    return f"An unexpected error occurred: {e}"
```

### Data Flow Implementation

1. **Input → Processing**
   - User input is received as JSON
   - Parsed and validated
   - Preferences extracted

2. **Analysis → Classification**
   - Goal analyzed using LLM
   - Classified using example labels
   - Decision made on breakdown possibility

3. **Task Generation → Output**
   - Tasks generated using LLM
   - Formatted as JSON
   - Validated and cleaned
   - Returned to user

This implementation ensures:
- Robust error handling at each step
- Clean JSON processing
- Proper integration with Cheshire Cat's LLM
- Consistent message formatting
- Reliable task breakdown generation 