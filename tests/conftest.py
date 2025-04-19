import os
import sys
import pytest
import json
from unittest.mock import MagicMock, patch

# Add the necessary directories to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
plugin_dir = os.path.dirname(current_dir)
core_dir = os.path.abspath(os.path.join(plugin_dir, '..', '..', '..'))

# Add the test directory to Python path to use our mock cat package
sys.path.insert(0, current_dir)
sys.path.insert(0, plugin_dir)
sys.path.insert(0, core_dir)

# Create a mock for the cat package structure
cat = MagicMock()
cat.log = MagicMock()
cat.log.log = MagicMock()
cat.mad_hatter = MagicMock()
cat.mad_hatter.decorators = MagicMock()

# Create a mock tool decorator that returns the function itself
def mock_tool(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

cat.mad_hatter.decorators.tool = mock_tool
cat.mad_hatter.decorators.hook = MagicMock()

# Mock the looking_glass module and its components
cat.looking_glass = MagicMock()
cat.looking_glass.output_parser = MagicMock()
cat.looking_glass.output_parser.LLMAction = MagicMock()

# Create a mock LLMAction class
class MockLLMAction:
    def __init__(self, *args, **kwargs):
        pass

cat.looking_glass.output_parser.LLMAction = MockLLMAction

# Add the mocks to sys.modules
sys.modules['cat'] = cat
sys.modules['cat.log'] = cat.log
sys.modules['cat.mad_hatter'] = cat.mad_hatter
sys.modules['cat.mad_hatter.decorators'] = cat.mad_hatter.decorators
sys.modules['cat.looking_glass'] = cat.looking_glass
sys.modules['cat.looking_glass.output_parser'] = cat.looking_glass.output_parser

@pytest.fixture(scope="session")
def cat_instance():
    """Create a mock Cheshire Cat instance for testing"""
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

@pytest.fixture
def sample_input():
    """Provide a standard valid input for testing"""
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
def mock_llm_response():
    """Provide a mock LLM response"""
    return {
        "tasks": {
            "name": "test",
            "duration": 30,
            "description": "test description"
        }
    }

@pytest.fixture
def mock_error_response():
    """Provide a mock error response"""
    return "Error: Invalid input" 