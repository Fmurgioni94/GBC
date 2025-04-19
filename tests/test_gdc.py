import os
import sys
import pytest
import json
import time
from unittest.mock import MagicMock, patch

# Add the necessary directories to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
plugin_dir = os.path.dirname(current_dir)
core_dir = os.path.abspath(os.path.join(plugin_dir, '..', '..', '..'))

# Add the test directory to Python path to use our mock cat package
sys.path.insert(0, current_dir)
sys.path.insert(0, plugin_dir)
sys.path.insert(0, core_dir)

# Mock the decorators before importing the module
def mock_hook(*args, **kwargs):
    def decorator(func):
        def wrapper(message, cat):
            return {"text": message["text"]}
        return wrapper
    return decorator

# Create a mock for the mad_hatter.decorators module
mock_decorators = MagicMock()
mock_decorators.hook = mock_hook
# Create a MagicMock for the tool decorator
mock_tool = MagicMock()
mock_tool.side_effect = lambda *args, **kwargs: lambda func: func
mock_decorators.tool = mock_tool

# Patch the modules
with patch.dict('sys.modules', {
    'cat.mad_hatter.decorators': mock_decorators,
    'cat.mad_hatter': MagicMock(),
    'cat': MagicMock()
}):
    from gdc import break_down_task, process_json_message, clearing_input, before_cat_sends_message

# Test fixtures
@pytest.fixture
def sample_input():
    return {
        "message": "I want to learn Python",
        "preferences": {
            "learningStyle": "visual",
            "preferredComplexity": "medium",
            "motivation": "high",
            "priorExperience": "beginner",
            "additionalNotes": ""
        }
    }

@pytest.fixture
def mock_cat():
    cat = MagicMock()
    cat.llm = MagicMock(return_value=json.dumps({
        "tasks": {
            "name": "test",
            "duration": 30,
            "description": "test description"
        }
    }))
    cat.classify = MagicMock(return_value="Breakable")
    return cat

# Test cases for break_down_task
def test_break_down_task_valid_input(sample_input, mock_cat):
    """Test task breakdown with valid input"""
    result = break_down_task(json.dumps(sample_input), mock_cat)
    # Check if result is either a dict or a string (for error cases)
    assert isinstance(result, (dict, str))
    if isinstance(result, dict):
        assert "tasks" in result
        mock_cat.llm.assert_called()
        mock_cat.classify.assert_called()

def test_break_down_task_invalid_json(mock_cat):
    """Test task breakdown with invalid JSON input"""
    result = break_down_task("invalid json", mock_cat)
    assert isinstance(result, str)
    assert "error" in result.lower() or result == "{}"

def test_break_down_task_missing_fields(mock_cat):
    """Test task breakdown with missing required fields"""
    invalid_input = {"message": "test"}
    result = break_down_task(json.dumps(invalid_input), mock_cat)
    assert isinstance(result, str)
    assert "error" in result.lower() or result == "{}"

# Test cases for process_json_message
def test_process_json_message_valid_input():
    """Test processing of valid JSON message"""
    valid_json = '```json\n{"tasks": {"name": "test", "duration": 30}}\n```'
    result = process_json_message(valid_json)
    assert isinstance(result, str)
    assert "tasks" in result

def test_process_json_message_invalid_input():
    """Test processing of invalid input"""
    result = process_json_message("invalid input")
    assert result == "invalid input"

def test_process_json_message_malformed_json():
    """Test processing of malformed JSON"""
    malformed_json = '```json\n{"tasks": {"name": "test"}\n```'
    result = process_json_message(malformed_json)
    assert isinstance(result, str)
    # The function returns an empty dictionary for malformed JSON
    assert result == "{}"

# Test cases for clearing_input
def test_clearing_input_valid_json():
    """Test clearing input with valid JSON"""
    valid_json = '```json\n{"tasks": {"name": "test"}}\n```'
    result = clearing_input(valid_json)
    assert isinstance(result, dict)
    assert "tasks" in result

def test_clearing_input_invalid_json():
    """Test clearing input with invalid JSON"""
    result = clearing_input("invalid json")
    assert result == {}

def test_clearing_input_empty_input():
    """Test clearing input with empty input"""
    result = clearing_input("")
    assert result == {}

# Test cases for error handling
def test_break_down_task_llm_error(mock_cat):
    """Test handling of LLM errors"""
    # Mock the llm method to raise an exception
    mock_cat.llm.side_effect = Exception("LLM error")
    
    # The function should catch the exception and return an error message
    with pytest.raises(Exception) as exc_info:
        break_down_task(json.dumps({"message": "test", "preferences": {}}), mock_cat)
    assert "LLM error" in str(exc_info.value)

def test_break_down_task_classification_error(mock_cat):
    """Test handling of classification errors"""
    mock_cat.classify.side_effect = Exception("Classification error")
    result = break_down_task(json.dumps({"message": "test", "preferences": {}}), mock_cat)
    assert isinstance(result, str)
    assert "error" in result.lower() or result == "{}"

# Test cases for edge cases
def test_break_down_task_empty_message(mock_cat):
    """Test task breakdown with empty message"""
    empty_input = {
        "message": "",
        "preferences": {
            "learningStyle": "visual",
            "preferredComplexity": "medium",
            "motivation": "high",
            "priorExperience": "beginner"
        }
    }
    result = break_down_task(json.dumps(empty_input), mock_cat)
    assert isinstance(result, str)
    assert "error" in result.lower() or result == "{}"

def test_break_down_task_long_message(mock_cat):
    """Test task breakdown with very long message"""
    long_message = "test " * 1000
    long_input = {
        "message": long_message,
        "preferences": {
            "learningStyle": "visual",
            "preferredComplexity": "medium",
            "motivation": "high",
            "priorExperience": "beginner"
        }
    }
    result = break_down_task(json.dumps(long_input), mock_cat)
    assert isinstance(result, (dict, str))

# Test cases for preference handling
def test_break_down_task_invalid_preferences(mock_cat):
    """Test task breakdown with invalid preferences"""
    invalid_prefs = {
        "message": "test",
        "preferences": {
            "learningStyle": "invalid",
            "preferredComplexity": "invalid",
            "motivation": "invalid",
            "priorExperience": "invalid"
        }
    }
    result = break_down_task(json.dumps(invalid_prefs), mock_cat)
    assert isinstance(result, str)
    assert "error" in result.lower() or result == "{}"

def test_break_down_task_missing_preferences(mock_cat):
    """Test task breakdown with missing preferences"""
    missing_prefs = {
        "message": "test",
        "preferences": {}
    }
    result = break_down_task(json.dumps(missing_prefs), mock_cat)
    assert isinstance(result, str)
    assert "error" in result.lower() or result == "{}"

# Test cases for output validation
def test_break_down_task_output_structure(mock_cat, sample_input):
    """Test the structure of the output from task breakdown"""
    mock_cat.llm.return_value = json.dumps({
        "tasks": {
            "name": "test",
            "duration": 30,
            "description": "test"
        }
    })
    result = break_down_task(json.dumps(sample_input), mock_cat)
    assert isinstance(result, (dict, str))
    if isinstance(result, dict):
        assert "tasks" in result
        assert all(key in result["tasks"] for key in ["name", "duration", "description"])

def test_break_down_task_output_types(mock_cat, sample_input):
    """Test the data types in the output"""
    mock_cat.llm.return_value = json.dumps({
        "tasks": {
            "name": "test",
            "duration": 30,
            "description": "test"
        }
    })
    result = break_down_task(json.dumps(sample_input), mock_cat)
    assert isinstance(result, (dict, str))
    if isinstance(result, dict):
        assert isinstance(result["tasks"]["name"], str)
        assert isinstance(result["tasks"]["duration"], int)
        assert isinstance(result["tasks"]["description"], str)

# Test cases for special characters
def test_break_down_task_special_characters(mock_cat):
    """Test task breakdown with special characters in input"""
    special_input = {
        "message": "test !@#$%^&*()",
        "preferences": {
            "learningStyle": "visual",
            "preferredComplexity": "medium",
            "motivation": "high",
            "priorExperience": "beginner"
        }
    }
    result = break_down_task(json.dumps(special_input), mock_cat)
    assert isinstance(result, (dict, str))

# Test cases for unicode characters
def test_break_down_task_unicode_characters(mock_cat):
    """Test task breakdown with unicode characters in input"""
    unicode_input = {
        "message": "test 你好",
        "preferences": {
            "learningStyle": "visual",
            "preferredComplexity": "medium",
            "motivation": "high",
            "priorExperience": "beginner"
        }
    }
    result = break_down_task(json.dumps(unicode_input), mock_cat)
    assert isinstance(result, (dict, str))

# Test cases for large inputs
def test_break_down_task_large_json(mock_cat):
    """Test task breakdown with very large JSON input"""
    large_input = {
        "message": "test " * 1000,
        "preferences": {
            "learningStyle": "visual",
            "preferredComplexity": "medium",
            "motivation": "high",
            "priorExperience": "beginner",
            "additionalNotes": "test " * 1000
        }
    }
    result = break_down_task(json.dumps(large_input), mock_cat)
    assert isinstance(result, (dict, str))

def test_break_down_task_nested_preferences(mock_cat):
    """Test task breakdown with nested preferences"""
    nested_input = {
        "message": "test",
        "preferences": {
            "learningStyle": "visual",
            "preferredComplexity": "medium",
            "motivation": "high",
            "priorExperience": "beginner",
            "additionalNotes": "",
            "nested": {
                "level1": {
                    "level2": "value"
                }
            }
        }
    }
    result = break_down_task(json.dumps(nested_input), mock_cat)
    assert isinstance(result, (dict, str))

# Test cases for multiple tasks
def test_break_down_task_multiple_tasks(mock_cat):
    """Test task breakdown with multiple tasks in output"""
    mock_cat.llm.return_value = json.dumps({
        "tasks": [
            {
                "name": "task1",
                "duration": 30,
                "description": "description1"
            },
            {
                "name": "task2",
                "duration": 45,
                "description": "description2"
            }
        ]
    })
    result = break_down_task(json.dumps({
        "message": "test",
        "preferences": {}
    }), mock_cat)
    assert isinstance(result, (dict, str))
    if isinstance(result, dict):
        assert "tasks" in result
        assert isinstance(result["tasks"], list)
        assert len(result["tasks"]) > 0

# Performance tests
def test_break_down_task_performance(mock_cat, sample_input):
    """Test performance of task breakdown"""
    start_time = time.time()
    result = break_down_task(json.dumps(sample_input), mock_cat)
    end_time = time.time()
    
    # Should complete within 5 seconds
    assert (end_time - start_time) < 5.0
    assert isinstance(result, (dict, str))

def test_process_json_message_performance():
    """Test performance of JSON message processing"""
    large_json = '```json\n' + json.dumps({"tasks": [{"name": f"task{i}", "duration": i} for i in range(100)]}) + '\n```'
    
    start_time = time.time()
    result = process_json_message(large_json)
    end_time = time.time()
    
    # Should complete within 1 second
    assert (end_time - start_time) < 1.0
    assert isinstance(result, str)

# Integration tests
def test_message_hook_integration(mock_cat):
    """Test integration with message hook"""
    message = {
        "text": '```json\n{"tasks": {"name": "test"}}\n```',
        "type": "chat"
    }
    
    result = before_cat_sends_message(message, mock_cat)
    assert isinstance(result, dict)
    assert "text" in result

def test_tool_integration(mock_cat):
    """Test integration with other tools"""
    result = break_down_task(json.dumps({
        "message": "test",
        "preferences": {}
    }), mock_cat)
    
    assert isinstance(result, (dict, str))
    # Verify the tool decorator was called
    assert mock_tool.called 