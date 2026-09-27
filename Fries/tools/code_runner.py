import os
import shutil
import subprocess
import tempfile
import uuid


LANGUAGE_ALIASES = {
    "py": "python",
    "python3": "python",
    "cpp": "cpp",
    "c++": "cpp",
    "js": "javascript",
    "node": "javascript",
    "ts": "typescript",
    "cs": "csharp",
    "c#": "csharp",
    "rb": "ruby",
    "rs": "rust",
    "kt": "kotlin",
    "sh": "bash",
    "shell": "bash",
    "yml": "yaml",
}

def normalize_language(language):
    language = language.lower().strip()
    return LANGUAGE_ALIASES.get(language, language)
def command_exists(command):
    return shutil.which(command) is not None
def run_process(command, cwd, timeout):
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr,
            "return_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": f"Program timed out after {timeout} seconds.",
            "return_code": -1
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "return_code": -1
        }
def run_python(code, directory, timeout):
    file_path = os.path.join(directory, "main.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    return run_process(
        ["python", file_path],
        directory,
        timeout
    )
def run_cpp(code, directory, timeout):
    source = os.path.join(directory, "main.cpp")
    executable = os.path.join(directory, "program.exe")
    with open(source, "w", encoding="utf-8") as f:
        f.write(code)
    if not command_exists("g++"):
        return {
            "success": False,
            "output": "",
            "error": "g++ is not installed or is not available in PATH.",
            "return_code": -1
        }
    compile_result = run_process(
        ["g++", source, "-o", executable],
        directory,
        timeout
    )
    if not compile_result["success"]:
        return {
            "success": False,
            "output": "",
            "error": "C++ compilation failed:\n" + compile_result["error"],
            "return_code": compile_result["return_code"]
        }
    return run_process(
        [executable],
        directory,
        timeout
    )
def run_c(code, directory, timeout):
    source = os.path.join(directory, "main.c")
    executable = os.path.join(directory, "program.exe")
    with open(source, "w", encoding="utf-8") as f:
        f.write(code)
    if not command_exists("gcc"):
        return {
            "success": False,
            "output": "",
            "error": "gcc is not installed or is not available in PATH.",
            "return_code": -1
        }
    compile_result = run_process(
        ["gcc", source, "-o", executable],
        directory,
        timeout
    )
    if not compile_result["success"]:
        return {
            "success": False,
            "output": "",
            "error": "C compilation failed:\n" + compile_result["error"],
            "return_code": compile_result["return_code"]
        }
    return run_process(
        [executable],
        directory,
        timeout
    )
def run_java(code, directory, timeout):
    class_name = "FriesProgram_" + uuid.uuid4().hex[:8]
    code = code.replace(
        "public class Main",
        f"public class {class_name}"
    )
    file_path = os.path.join(
        directory,
        f"{class_name}.java"
    )
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    if not command_exists("javac"):
        return {
            "success": False,
            "output": "",
            "error": "javac is not installed or is not available in PATH.",
            "return_code": -1
        }
    compile_result = run_process(
        ["javac", file_path],
        directory,
        timeout
    )
    if not compile_result["success"]:
        return {
            "success": False,
            "output": "",
            "error": "Java compilation failed:\n" + compile_result["error"],
            "return_code": compile_result["return_code"]
        }
    return run_process(
        ["java", "-cp", directory, class_name],
        directory,
        timeout
    )
def run_javascript(code, directory, timeout):
    file_path = os.path.join(directory, "main.js")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    if not command_exists("node"):
        return {
            "success": False,
            "output": "",
            "error": "Node.js is not installed or is not available in PATH.",
            "return_code": -1
        }
    return run_process(
        ["node", file_path],
        directory,
        timeout
    )
def run_typescript(code, directory, timeout):
    file_path = os.path.join(directory, "main.ts")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    if not command_exists("ts-node"):
        return {
            "success": False,
            "output": "",
            "error": "ts-node is not installed or is not available in PATH.",
            "return_code": -1
        }
    return run_process(
        ["ts-node", file_path],
        directory,
        timeout
    )
def run_php(code, directory, timeout):
    file_path = os.path.join(directory, "main.php")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    if not command_exists("php"):
        return {
            "success": False,
            "output": "",
            "error": "PHP is not installed or is not available in PATH.",
            "return_code": -1
        }
    return run_process(
        ["php", file_path],
        directory,
        timeout
    )
def run_ruby(code, directory, timeout):
    file_path = os.path.join(directory, "main.rb")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    if not command_exists("ruby"):
        return {
            "success": False,
            "output": "",
            "error": "Ruby is not installed or is not available in PATH.",
            "return_code": -1
        }
    return run_process(
        ["ruby", file_path],
        directory,
        timeout
    )
def run_go(code, directory, timeout):
    file_path = os.path.join(directory, "main.go")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    if not command_exists("go"):
        return {
            "success": False,
            "output": "",
            "error": "Go is not installed or is not available in PATH.",
            "return_code": -1
        }
    return run_process(
        ["go", "run", file_path],
        directory,
        timeout
    )
def run_rust(code, directory, timeout):
    source = os.path.join(directory, "main.rs")
    executable = os.path.join(directory, "program.exe")
    with open(source, "w", encoding="utf-8") as f:
        f.write(code)
    if not command_exists("rustc"):
        return {
            "success": False,
            "output": "",
            "error": "Rust (rustc) is not installed or is not available in PATH.",
            "return_code": -1
        }
    compile_result = run_process(
        ["rustc", source, "-o", executable],
        directory,
        timeout
    )
    if not compile_result["success"]:
        return {
            "success": False,
            "output": "",
            "error": "Rust compilation failed:\n" + compile_result["error"],
            "return_code": compile_result["return_code"]
        }
    return run_process(
        [executable],
        directory,
        timeout
    )
def run_csharp(code, directory, timeout):
    file_path = os.path.join(directory, "Program.cs")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    if not command_exists("dotnet"):
        return {
            "success": False,
            "output": "",
            "error": ".NET SDK is not installed or is not available in PATH.",
            "return_code": -1
        }
    project_result = run_process(
        [
            "dotnet",
            "new",
            "console",
            "--force",
            "--no-restore"
        ],
        directory,
        timeout
    )
    if not project_result["success"]:
        return {
            "success": False,
            "output": "",
            "error": project_result["error"],
            "return_code": project_result["return_code"]
        }
    return run_process(
        ["dotnet", "run", "--no-restore"],
        directory,
        timeout
    )
def run_kotlin(code, directory, timeout):
    file_path = os.path.join(directory, "Main.kt")
    jar_file = os.path.join(directory, "program.jar")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    if not command_exists("kotlinc"):
        return {
            "success": False,
            "output": "",
            "error": "Kotlin compiler (kotlinc) is not installed or is not available in PATH.",
            "return_code": -1
        }
    compile_result = run_process(
        ["kotlinc", file_path, "-include-runtime", "-d", jar_file],
        directory,
        timeout
    )
    if not compile_result["success"]:
        return {
            "success": False,
            "output": "",
            "error": "Kotlin compilation failed:\n" + compile_result["error"],
            "return_code": compile_result["return_code"]
        }
    if not command_exists("java"):
        return {
            "success": False,
            "output": "",
            "error": "Java is required to run the Kotlin program.",
            "return_code": -1
        }
    return run_process(
        ["java", "-jar", jar_file],
        directory,
        timeout
    )
def run_bash(code, directory, timeout):
    file_path = os.path.join(directory, "main.sh")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    if not command_exists("bash"):
        return {
            "success": False,
            "output": "",
            "error": "Bash is not installed or is not available in PATH.",
            "return_code": -1
        }
    return run_process(
        ["bash", file_path],
        directory,
        timeout
    )
RUNNERS = {
    "python": run_python,
    "cpp": run_cpp,
    "c": run_c,
    "java": run_java,
    "javascript": run_javascript,
    "typescript": run_typescript,
    "php": run_php,
    "ruby": run_ruby,
    "go": run_go,
    "rust": run_rust,
    "csharp": run_csharp,
    "kotlin": run_kotlin,
    "bash": run_bash,
}
def run_code(language, code, timeout=5):
    language = normalize_language(language)
    if language not in RUNNERS:
        return {
            "success": False,
            "language": language,
            "output": "",
            "error": (
                f"Language '{language}' is not currently supported. "
                f"Supported languages: {', '.join(RUNNERS.keys())}"
            ),
            "return_code": -1
        }
    if not isinstance(code, str) or not code.strip():
        return {
            "success": False,
            "language": language,
            "output": "",
            "error": "No code was provided.",
            "return_code": -1
        }
    try:
        with tempfile.TemporaryDirectory(
            prefix="fries_code_"
        ) as directory:
            result = RUNNERS[language](
                code,
                directory,
                timeout
            )
            result["language"] = language
            return result
    except Exception as e:
        return {
            "success": False,
            "language": language,
            "output": "",
            "error": str(e),
            "return_code": -1
        }