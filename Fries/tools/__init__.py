from .calculator import calculate
from .weather import get_weather
from .unit_converter import unit_converter
from .currency_converter import currency_converter
from .datetime_tool import time_tool
from .ocr_tool import extract_text_from_image

from .file_reader import (
    read_file,
    write_file,
    append_file,
    list_files,
    delete_file,
    copy_file,
    move_file,
    get_file_info
)

from .pdf_tool import (
    create_pdf,
    merge_pdfs,
    split_pdf,
    extract_pages,
    extract_text,
    delete_pages,
    rotate_pages,
    get_pdf_info
)

from .knowledge_search import knowledge_search
from .code_runner import run_code


TOOLS = {
    "calculator": calculate,
    "get_weather": get_weather,
    "unit_converter": unit_converter,
    "currency_converter": currency_converter,
    "time_tool": time_tool,

    "read_file": read_file,
    "write_file": write_file,
    "append_file": append_file,
    "list_files": list_files,
    "delete_file": delete_file,
    "copy_file": copy_file,
    "move_file": move_file,
    "get_file_info": get_file_info,

    "pdf_create": create_pdf,
    "pdf_merge": merge_pdfs,
    "pdf_split": split_pdf,
    "pdf_extract_pages": extract_pages,
    "pdf_extract_text": extract_text,
    "pdf_delete_pages": delete_pages,
    "pdf_rotate_pages": rotate_pages,
    "pdf_info": get_pdf_info,

    "knowledge_search": knowledge_search,
    "code_runner": run_code,

    "extract_text_from_image": extract_text_from_image,
}


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "name": "calculator",
        "description": "Perform mathematical calculations.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to calculate."
                }
            },
            "required": ["expression"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather information for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or location name."
                }
            },
            "required": ["location"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "unit_converter",
        "description": "Convert values between different units.",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "Value to convert."
                },
                "from_unit": {
                    "type": "string",
                    "description": "Unit to convert from."
                },
                "to_unit": {
                    "type": "string",
                    "description": "Unit to convert to."
                }
            },
            "required": ["value", "from_unit", "to_unit"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "currency_converter",
        "description": "Convert an amount from one currency to another.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "Amount to convert."
                },
                "from_currency": {
                    "type": "string",
                    "description": "Currency to convert from."
                },
                "to_currency": {
                    "type": "string",
                    "description": "Currency to convert to."
                }
            },
            "required": ["amount", "from_currency", "to_currency"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "time_tool",
        "description": "Get the current date and time.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone name."
                }
            },
            "required": [],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "read_file",
        "description": "Read the contents of a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Full path of the file."
                }
            },
            "required": ["file_path"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "write_file",
        "description": "Write content to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Full path of the file."
                },
                "content": {
                    "type": "string",
                    "description": "Content to write."
                }
            },
            "required": ["file_path", "content"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "append_file",
        "description": "Append content to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Full path of the file."
                },
                "content": {
                    "type": "string",
                    "description": "Content to append."
                }
            },
            "required": ["file_path", "content"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "list_files",
        "description": "List files in a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path."
                }
            },
            "required": ["directory"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "delete_file",
        "description": "Delete a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Full path of the file."
                }
            },
            "required": ["file_path"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "copy_file",
        "description": "Copy a file to another location.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source file path."
                },
                "destination": {
                    "type": "string",
                    "description": "Destination file path."
                }
            },
            "required": ["source", "destination"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "move_file",
        "description": "Move a file to another location.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source file path."
                },
                "destination": {
                    "type": "string",
                    "description": "Destination file path."
                }
            },
            "required": ["source", "destination"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "get_file_info",
        "description": "Get information about a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Full path of the file."
                }
            },
            "required": ["file_path"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "pdf_create",
        "description": "Create a PDF file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Output PDF path."
                },
                "content": {
                    "type": "string",
                    "description": "Text content for the PDF."
                }
            },
            "required": ["file_path", "content"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "pdf_merge",
        "description": "Merge multiple PDF files into one PDF.",
        "parameters": {
            "type": "object",
            "properties": {
                "input_files": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of PDF files to merge."
                },
                "output_file": {
                    "type": "string",
                    "description": "Output PDF path."
                }
            },
            "required": ["input_files", "output_file"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "pdf_split",
        "description": "Split a PDF into separate PDF files.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "PDF file path."
                },
                "output_directory": {
                    "type": "string",
                    "description": "Directory where split PDFs should be saved."
                }
            },
            "required": ["file_path", "output_directory"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "pdf_extract_pages",
        "description": "Extract selected pages from a PDF.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "PDF file path."
                },
                "pages": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "Page numbers to extract."
                },
                "output_file": {
                    "type": "string",
                    "description": "Output PDF path."
                }
            },
            "required": ["file_path", "pages", "output_file"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "pdf_extract_text",
        "description": "Extract text from a PDF.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "PDF file path."
                }
            },
            "required": ["file_path"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "pdf_delete_pages",
        "description": "Delete selected pages from a PDF.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "PDF file path."
                },
                "pages": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "Page numbers to delete."
                },
                "output_file": {
                    "type": "string",
                    "description": "Output PDF path."
                }
            },
            "required": ["file_path", "pages", "output_file"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "pdf_rotate_pages",
        "description": "Rotate pages in a PDF.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "PDF file path."
                },
                "pages": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "Page numbers to rotate."
                },
                "angle": {
                    "type": "integer",
                    "description": "Rotation angle."
                },
                "output_file": {
                    "type": "string",
                    "description": "Output PDF path."
                }
            },
            "required": ["file_path", "pages", "angle", "output_file"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "pdf_info",
        "description": "Get information about a PDF.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "PDF file path."
                }
            },
            "required": ["file_path"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "knowledge_search",
        "description": "Search the local knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "code_runner",
        "description": "Run code in a supported programming language.",
        "parameters": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "description": "Programming language."
                },
                "code": {
                    "type": "string",
                    "description": "Code to execute."
                }
            },
            "required": ["language", "code"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "extract_text_from_image",
        "description": "Extract text from an image using OCR.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Full path of the image file."
                }
            },
            "required": ["file_path"],
            "additionalProperties": False
        }
    }
]