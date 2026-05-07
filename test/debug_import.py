# debug_function_tool.py
from google.adk.tools import FunctionTool
import inspect

print("FunctionTool __init__ signature:")
print(inspect.signature(FunctionTool.__init__))

# Check what attributes FunctionTool has
tool = FunctionTool(lambda x: x)
print("\nFunctionTool attributes:")
print(dir(tool))