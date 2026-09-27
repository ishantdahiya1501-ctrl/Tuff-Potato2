import os
import shutil
import mimetypes
import tkinter as tk
from tkinter import scrolledtext

TEXT_EXTENSIONS = {
    ".txt",".md",".py",".js",".ts",".jsx",".tsx",".html",".css",".scss",
    ".json",".xml",".yaml",".yml",".csv",".tsv",".log",".ini",".cfg",
    ".conf",".env",".bat",".cmd",".ps1",".sh",".sql",".c",".cpp",".h",
    ".hpp",".java",".cs",".go",".rs",".php",".rb",".swift",".kt",".dart"
}

def ask_permission(action, file_path, details=""):
    result = {"allowed": False}

    def allow():
        result["allowed"] = True
        window.destroy()

    def reject():
        result["allowed"] = False
        window.destroy()

    window = tk.Tk()
    window.title("FlexAI Permission")
    window.geometry("620x500")
    window.configure(bg="#111111")
    window.resizable(False, False)

    outer = tk.Frame(
        window,
        bg="#111111"
    )
    outer.pack(
        fill="both",
        expand=True,
        padx=25,
        pady=25
    )

    title = tk.Label(
        outer,
        text="FlexAI wants permission",
        font=("Segoe UI", 18, "bold"),
        fg="white",
        bg="#111111"
    )
    title.pack(
        anchor="w",
        pady=(5, 4)
    )

    subtitle = tk.Label(
        outer,
        text="A file operation requires your approval.",
        font=("Segoe UI", 10),
        fg="#999999",
        bg="#111111"
    )
    subtitle.pack(
        anchor="w",
        pady=(0, 18)
    )

    info = tk.Frame(
        outer,
        bg="#242424",
        highlightthickness=1,
        highlightbackground="#333333"
    )
    info.pack(
        fill="both",
        expand=True
    )

    action_label = tk.Label(
        info,
        text=f"ACTION\n{action.upper()}",
        font=("Segoe UI", 10, "bold"),
        fg="#bbbbbb",
        bg="#242424",
        justify="left"
    )
    action_label.pack(
        anchor="w",
        padx=18,
        pady=(18, 8)
    )

    file_label = tk.Label(
        info,
        text=f"FILE\n{file_path}",
        font=("Segoe UI", 10),
        fg="white",
        bg="#242424",
        justify="left",
        wraplength=530
    )
    file_label.pack(
        anchor="w",
        padx=18,
        pady=(0, 12)
    )

    if details:
        details_label = tk.Label(
            info,
            text="DETAILS",
            font=("Segoe UI", 10, "bold"),
            fg="#bbbbbb",
            bg="#242424"
        )
        details_label.pack(
            anchor="w",
            padx=18,
            pady=(0, 6)
        )

        details_box = scrolledtext.ScrolledText(
            info,
            height=10,
            bg="#181818",
            fg="#dddddd",
            insertbackground="white",
            relief="flat",
            borderwidth=0,
            font=("Consolas", 9),
            wrap=tk.WORD
        )

        details_box.insert(
            "1.0",
            details
        )

        details_box.config(
            state="disabled"
        )

        details_box.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=(0, 18)
        )

    buttons = tk.Frame(
        outer,
        bg="#111111"
    )
    buttons.pack(
        fill="x",
        pady=(20, 0)
    )

    reject_button = tk.Button(
        buttons,
        text="Reject",
        command=reject,
        font=("Segoe UI", 10, "bold"),
        fg="white",
        bg="#d9363e",
        activebackground="#b82d34",
        activeforeground="white",
        relief="flat",
        borderwidth=0,
        cursor="hand2",
        width=16,
        height=2
    )
    reject_button.pack(
        side="left",
        padx=(0, 10)
    )

    allow_button = tk.Button(
        buttons,
        text="Allow",
        command=allow,
        font=("Segoe UI", 10, "bold"),
        fg="white",
        bg="#2878f0",
        activebackground="#1d5fc4",
        activeforeground="white",
        relief="flat",
        borderwidth=0,
        cursor="hand2",
        width=16,
        height=2
    )
    allow_button.pack(
        side="right",
        padx=(10, 0)
    )

    window.protocol(
        "WM_DELETE_WINDOW",
        reject
    )

    window.mainloop()

    return result["allowed"]
def get_file_info(file_path):
    if not os.path.exists(file_path):
        return {
            "success": False,
            "error": f"File does not exist: {file_path}"
        }

    if not os.path.isfile(file_path):
        return {
            "success": False,
            "error": f"Path is not a file: {file_path}"
        }

    size = os.path.getsize(file_path)
    extension = os.path.splitext(file_path)[1].lower()
    mime_type, _ = mimetypes.guess_type(file_path)

    return {
        "success": True,
        "file": os.path.abspath(file_path),
        "name": os.path.basename(file_path),
        "extension": extension,
        "size_bytes": size,
        "mime_type": mime_type
    }
def read_file(file_path):
    if not os.path.exists(file_path):
        return {
            "success": False,
            "error": f"File does not exist: {file_path}"
        }

    if not os.path.isfile(file_path):
        return {
            "success": False,
            "error": f"Path is not a file: {file_path}"
        }

    extension = os.path.splitext(file_path)[1].lower()

    try:
        if extension in TEXT_EXTENSIONS:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            return {
                "success": True,
                "operation": "read",
                "file": os.path.abspath(file_path),
                "type": "text",
                "content": content
            }

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            return {
                "success": True,
                "operation": "read",
                "file": os.path.abspath(file_path),
                "type": "text",
                "content": content
            }

        except UnicodeDecodeError:
            pass

        info = get_file_info(file_path)

        return {
            "success": True,
            "operation": "read",
            "file": os.path.abspath(file_path),
            "type": "binary",
            "message": "This is a binary file and cannot be returned as normal text.",
            "file_info": info
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
def write_file(file_path, content):
    file_exists = os.path.exists(file_path)

    action = "edit" if file_exists else "create"

    details = (
        f"Path:\n{os.path.abspath(file_path)}\n\n"
        f"Operation:\n{'Overwrite existing file' if file_exists else 'Create new file'}\n\n"
        f"Content:\n{content}"
    )

    if not ask_permission(action, file_path, details):
        return {
            "success": False,
            "operation": action,
            "file": os.path.abspath(file_path),
            "message": "User rejected the operation."
        }

    try:
        directory = os.path.dirname(os.path.abspath(file_path))

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)

        return {
            "success": True,
            "operation": action,
            "file": os.path.abspath(file_path),
            "message": "File written successfully."
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
def append_file(file_path, content):
    details = (
        f"Path:\n{os.path.abspath(file_path)}\n\n"
        f"Operation:\nAppend content\n\n"
        f"Content to append:\n{content}"
    )

    if not ask_permission("append", file_path, details):
        return {
            "success": False,
            "operation": "append",
            "file": os.path.abspath(file_path),
            "message": "User rejected the operation."
        }

    try:
        with open(file_path, "a", encoding="utf-8") as file:
            file.write(content)

        return {
            "success": True,
            "operation": "append",
            "file": os.path.abspath(file_path),
            "message": "Content appended successfully."
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
def list_files(directory="."):
    if not os.path.exists(directory):
        return {
            "success": False,
            "error": f"Directory does not exist: {directory}"
        }

    if not os.path.isdir(directory):
        return {
            "success": False,
            "error": f"Not a directory: {directory}"
        }

    try:
        items = []

        for name in os.listdir(directory):
            path = os.path.join(directory, name)

            if os.path.isfile(path):
                items.append({
                    "name": name,
                    "type": "file",
                    "size_bytes": os.path.getsize(path)
                })

            elif os.path.isdir(path):
                items.append({
                    "name": name,
                    "type": "directory"
                })

        return {
            "success": True,
            "operation": "list",
            "directory": os.path.abspath(directory),
            "items": items
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
def delete_file(file_path):
    details = (
        f"Path:\n{os.path.abspath(file_path)}\n\n"
        f"Operation:\nPERMANENTLY DELETE THIS FILE"
    )

    if not ask_permission("delete", file_path, details):
        return {
            "success": False,
            "operation": "delete",
            "file": os.path.abspath(file_path),
            "message": "User rejected the operation."
        }

    if not os.path.exists(file_path):
        return {
            "success": False,
            "error": f"File does not exist: {file_path}"
        }

    if not os.path.isfile(file_path):
        return {
            "success": False,
            "error": f"Not a file: {file_path}"
        }

    try:
        os.remove(file_path)

        return {
            "success": True,
            "operation": "delete",
            "file": os.path.abspath(file_path),
            "message": "File deleted successfully."
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
def copy_file(source, destination):
    details = (
        f"Source:\n{os.path.abspath(source)}\n\n"
        f"Destination:\n{os.path.abspath(destination)}\n\n"
        f"Operation:\nCopy file"
    )

    if not ask_permission("copy", destination, details):
        return {
            "success": False,
            "operation": "copy",
            "message": "User rejected the operation."
        }

    if not os.path.isfile(source):
        return {
            "success": False,
            "error": f"Source file does not exist: {source}"
        }

    try:
        directory = os.path.dirname(os.path.abspath(destination))

        if directory:
            os.makedirs(directory, exist_ok=True)

        shutil.copy2(source, destination)

        return {
            "success": True,
            "operation": "copy",
            "source": os.path.abspath(source),
            "destination": os.path.abspath(destination),
            "message": "File copied successfully."
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
def move_file(source, destination):
    details = (
        f"Source:\n{os.path.abspath(source)}\n\n"
        f"Destination:\n{os.path.abspath(destination)}\n\n"
        f"Operation:\nMove or rename file"
    )

    if not ask_permission("move", destination, details):
        return {
            "success": False,
            "operation": "move",
            "message": "User rejected the operation."
        }

    if not os.path.isfile(source):
        return {
            "success": False,
            "error": f"Source file does not exist: {source}"
        }

    try:
        directory = os.path.dirname(os.path.abspath(destination))

        if directory:
            os.makedirs(directory, exist_ok=True)

        shutil.move(source, destination)

        return {
            "success": True,
            "operation": "move",
            "source": os.path.abspath(source),
            "destination": os.path.abspath(destination),
            "message": "File moved successfully."
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    if not os.path.isfile(source):
        return {
            "success": False,
            "error": f"Source file does not exist: {source}"
        }
    try:
        directory = os.path.dirname(os.path.abspath(destination))
        if directory:
            os.makedirs(directory, exist_ok=True)
        shutil.move(source, destination)
        return {
            "success": True,
            "operation": "move",
            "source": os.path.abspath(source),
            "destination": os.path.abspath(destination),
            "message": "File moved successfully."
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }