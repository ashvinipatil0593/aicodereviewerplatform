import io
import json
import os
import re
import subprocess
import tempfile
from contextlib import redirect_stdout
from typing import Any, Dict, List

import requests

from deep_learning_module import (
    add_training_example,
    build_dl_insights,
    merge_learned_suggestions,
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("MODEL_NAME", "phi3")


def extract_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response")

    brace_count = 0
    end = -1

    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break

    if end != -1:
        return text[start:end]

    raise ValueError("Incomplete JSON object found in model response")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def clean_suggestions(suggestions: List[str]) -> List[str]:
    banned = [
        "run ollama",
        "ollama serve",
        "ollama pull",
        "make sure ollama is running",
        "__future__",
        "python 2",
        "shebang",
    ]

    cleaned = []
    seen = set()

    for s in suggestions:
        text = str(s).strip()

        while text.lower().startswith("learned pattern:"):
            text = text[len("learned pattern:"):].strip()

        if not text:
            continue

        lower = text.lower()
        if any(b in lower for b in banned):
            continue

        if lower not in seen:
            seen.add(lower)
            cleaned.append(text)

    return cleaned


def is_simple_python_code(code: str) -> bool:
    stripped = code.strip()
    lines = [line for line in stripped.splitlines() if line.strip()]
    return len(lines) <= 5 and len(stripped) < 250


def is_simple_js_code(code: str) -> bool:
    return len(code.strip()) < 600


def is_simple_java_code(code: str) -> bool:
    stripped = code.strip()
    return len(stripped) < 900 and "public class Main" in stripped


def get_model_options(language: str) -> Dict[str, Any]:
    lang = language.lower()

    if lang == "javascript":
        return {"temperature": 0.05, "num_predict": 180}

    if lang == "java":
        return {"temperature": 0.05, "num_predict": 220}

    if lang == "python":
        return {"temperature": 0.1, "num_predict": 350}

    return {"temperature": 0.1, "num_predict": 300}


def trim_code_for_review(code: str, max_chars: int = 2500) -> str:
    code = code.strip()
    if len(code) <= max_chars:
        return code
    return code[:max_chars] + "\n// truncated for faster review"


def generate_fixed_code(language: str, code: str, errors: list) -> str:
    fixed = code
    err_text = str(errors).lower()
    lang = language.lower()

    if lang == "python":
        fixed = re.sub(
            r"^(for .+|if .+|while .+|def .+)(?<!:)$",
            r"\1:",
            fixed,
            flags=re.MULTILINE,
        )

        if "was never closed" in err_text or "'(' was never closed" in err_text:
            open_count = fixed.count("(")
            close_count = fixed.count(")")
            if open_count > close_count:
                fixed += ")" * (open_count - close_count)

        if "unterminated string literal" in err_text:
            if fixed.count('"') % 2 != 0:
                fixed += '"'
            if fixed.count("(") > fixed.count(")"):
                fixed += ")"

        if "division by zero" in err_text:
            fixed = fixed.replace(
                "return total / len(numbers)",
                "return total / len(numbers) if len(numbers) > 0 else 0",
            )

    elif lang == "javascript":
        if "infinite loop" in err_text and "i++" not in fixed:
            fixed = fixed.replace("console.log(i);", "console.log(i);\n  i++;")

        if "response.json()" in fixed and "await response.json()" not in fixed:
            fixed = fixed.replace("response.json()", "await response.json()")

        if "i.age" in fixed:
            fixed = fixed.replace("i.age", "users[i].age")

    elif lang == "java":
        if "outside of methods" in err_text or (
            "public class" not in fixed and "class " not in fixed
        ):
            fixed = f"""public class Main {{
    public static void main(String[] args) {{
        {code}
    }}
}}"""

        if "infinite loop" in err_text and "i++" not in fixed:
            fixed = fixed.replace(
                "System.out.println(i);",
                "System.out.println(i);\n            i++;",
            )

        fixed = fixed.replace("<= arr.length", "< arr.length")
        fixed = fixed.replace("<= numbers.length", "< numbers.length")

    elif lang == "php":
        if "division by zero" in err_text:
            fixed = fixed.replace("/ 0", "/ 1")

        if "infinite loop" in err_text and "$i++" not in fixed:
            fixed = fixed.replace("echo $i;", "echo $i;\n   $i++;")

    elif lang in ["csharp", "c#", "dotnet"]:
        if "division by zero" in err_text:
            fixed = fixed.replace("/ 0", "/ 1")

        if "infinite loop" in err_text and "i++" not in fixed:
            fixed = fixed.replace(
                "Console.WriteLine(i);",
                "Console.WriteLine(i);\n            i++;"
            )

    elif lang in ["sql", "sqlserver"]:
        if "delete" in fixed.lower() and "where" not in fixed.lower():
            fixed = fixed.replace(
                "DELETE FROM",
                "-- Add WHERE condition\nDELETE FROM"
            )

    elif lang in ["html", "bootstrap"]:
        if "<html>" not in fixed.lower():
            fixed = "<html>\n<body>\n" + fixed + "\n</body>\n</html>"

    if fixed.strip() == code.strip() and errors:
        return "Could not safely auto-fix. See suggestions."

    return fixed

def default_response(code: str, message: str, language: str) -> Dict[str, Any]:
    errors = [
        {
            "line": 0,
            "message": message,
            "fix": "Check backend logs or try smaller code.",
        }
    ]

    return {
        "errors": errors,
        "suggestions": [],
        "output": "Could not predict output.",
        "fixedCode": generate_fixed_code(language, code, errors),
        "score": {
            "readability": 40,
            "performance": 40,
            "maintainability": 40,
            "security": 40,
        },
        "dlInsights": fast_dl_status(),
    }


def build_prompt(language: str, code: str) -> str:
    code = trim_code_for_review(code)

    return f"""
You are an expert {language} code reviewer.

Return STRICT VALID JSON only.
No markdown. No backticks. No extra text.
Use double quotes for all JSON keys and strings.
Keep response short.

Required JSON:
{{
  "errors": [
    {{
      "line": 0,
      "message": "string",
      "fix": "string"
    }}
  ],
  "suggestions": ["string"],
  "output": "string",
  "fixedCode": "string",
  "score": {{
    "readability": 0,
    "performance": 0,
    "maintainability": 0,
    "security": 0
  }}
}}

Rules:
1. If no major errors, return empty errors array.
2. Suggestions must be short and code-related only.
3. fixedCode must correct the code only when safe.
4. If code is already correct, fixedCode must be exactly same as input.
5. Do not give system setup suggestions.
6. Stop after final JSON brace.

Code:
{code}
""".strip()


def get_python_output(code: str) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as f:
            f.write(code)
            path = f.name

        result = subprocess.run(["python", path], capture_output=True, text=True, timeout=2)

        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"

        return result.stdout.strip() or "No output"

    except subprocess.TimeoutExpired:
        if "while" in code or "for" in code:
            return "Error: Possible infinite loop detected."
        else:
            return "Error: Program execution timed out."        
            
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        try:
            if "path" in locals() and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def get_js_output(code: str) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".js", mode="w", encoding="utf-8") as f:
            f.write(code)
            path = f.name

        result = subprocess.run(["node", path], capture_output=True, text=True, timeout=2)

        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"

        return result.stdout.strip() or "No output"

    except subprocess.TimeoutExpired:
        return "Error: Possible infinite loop detected."
    except FileNotFoundError:
        return "Cannot be determined safely."
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        try:
            if "path" in locals() and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def get_java_output(code: str) -> str:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "Main.java")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

            compile_result = subprocess.run(["javac", file_path], capture_output=True, text=True)

            if compile_result.returncode != 0:
                return "Cannot be determined safely."

            run_result = subprocess.run(
                ["java", "-cp", temp_dir, "Main"],
                capture_output=True,
                text=True,
                timeout=2,
            )

            if run_result.returncode != 0:
                return f"Error: {run_result.stderr.strip()}"

            return run_result.stdout.strip() or "No output"

    except subprocess.TimeoutExpired:
        return "Error: Possible infinite loop detected."
    except Exception as e:
        return f"Error: {str(e)}"


def run_c(code: str) -> str:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "main.c")
            exe_path = os.path.join(temp_dir, "main.exe")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

            compile_result = subprocess.run(
                ["gcc", file_path, "-o", exe_path],
                capture_output=True,
                text=True
            )

            if compile_result.returncode != 0:
                return f"Error: {compile_result.stderr.strip()}"

            run_result = subprocess.run(
                [exe_path],
                capture_output=True,
                text=True,
                timeout=2
            )

            if run_result.returncode != 0:
                return f"Error: {run_result.stderr.strip()}"

            return run_result.stdout.strip() or "No output"

    except subprocess.TimeoutExpired:
        return "Error: Possible infinite loop detected."

    except Exception as e:
        return f"Error: {str(e)}"
def run_cpp(code: str) -> str:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "main.cpp")
            exe_path = os.path.join(temp_dir, "main.exe")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

            compile_result = subprocess.run(
                ["g++", file_path, "-o", exe_path],
                capture_output=True,
                text=True
            )

            if compile_result.returncode != 0:
                return f"Error: {compile_result.stderr.strip()}"

            run_result = subprocess.run(
                [exe_path],
                capture_output=True,
                text=True,
                timeout=2
            )

            if run_result.returncode != 0:
                return f"Error: {run_result.stderr.strip()}"

            return run_result.stdout.strip() or "No output"

    except subprocess.TimeoutExpired:
        return "Error: Possible infinite loop detected."

    except Exception as e:
        return f"Error: {str(e)}"


def get_python_syntax_errors(code: str) -> List[Dict[str, Any]]:
    try:
        compile(code, "<string>", "exec")
        return []
    except SyntaxError as e:
        return [
            {
                "line": e.lineno or 0,
                "message": e.msg,
                "fix": "Fix the Python syntax error on this line.",
            }
        ]


def check_js_syntax(code: str) -> List[Dict[str, Any]]:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".js", mode="w", encoding="utf-8") as f:
            f.write(code)
            path = f.name

        result = subprocess.run(["node", "--check", path], capture_output=True, text=True)

        if result.returncode != 0:
            return [
                {
                    "line": 0,
                    "message": result.stderr.strip() or "JavaScript syntax error",
                    "fix": "Fix JavaScript syntax issue.",
                }
            ]

        return []

    except FileNotFoundError:
        return [
            {
                "line": 0,
                "message": "Node.js is not installed or not added to PATH.",
                "fix": "Install Node.js and restart terminal.",
            }
        ]
    except Exception as e:
        return [{"line": 0, "message": str(e), "fix": "Check JavaScript setup."}]
    finally:
        try:
            if "path" in locals() and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def check_java_code(code: str) -> List[Dict[str, Any]]:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "Main.java")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

            result = subprocess.run(["javac", file_path], capture_output=True, text=True)

            if result.returncode != 0:
                error_msg = result.stderr.strip() or "Java compilation error"

                if "outside of methods" in error_msg:
                    error_msg = "Code must be inside a class and main method in Java."

                return [
                    {
                        "line": 0,
                        "message": error_msg,
                        "fix": "Fix Java compilation issue. Use public class Main with main() method.",
                    }
                ]

        return []

    except FileNotFoundError:
        return [
            {
                "line": 0,
                "message": "javac is not installed or not added to PATH.",
                "fix": "Install JDK and restart terminal.",
            }
        ]
    except Exception as e:
        return [{"line": 0, "message": str(e), "fix": "Check Java setup."}]


def check_c_code(code: str) -> List[Dict[str, Any]]:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "main.c")
            output_path = os.path.join(temp_dir, "main.exe")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

            result = subprocess.run(
                ["gcc", file_path, "-o", output_path],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return [{
                    "line": 0,
                    "message": result.stderr.strip() or "C compilation error",
                    "fix": "Fix C syntax/compilation issue."
                }]

        return []

    except FileNotFoundError:
        return [{
            "line": 0,
            "message": "gcc compiler not found.",
            "fix": "Install GCC/MinGW and add it to PATH."
        }]

    except Exception as e:
        return [{
            "line": 0,
            "message": str(e),
            "fix": "Check C setup."
        }]
def check_cpp_code(code: str) -> List[Dict[str, Any]]:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "main.cpp")
            output_path = os.path.join(temp_dir, "main.exe")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

            result = subprocess.run(
                ["g++", file_path, "-o", output_path],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return [{
                    "line": 0,
                    "message": result.stderr.strip() or "C++ compilation error",
                    "fix": "Fix C++ syntax/compilation issue."
                }]

        return []

    except FileNotFoundError:
        return [{
            "line": 0,
            "message": "g++ compiler not found.",
            "fix": "Install MinGW/g++ and add it to PATH."
        }]

    except Exception as e:
        return [{
            "line": 0,
            "message": str(e),
            "fix": "Check C++ setup."
        }]


def check_static_errors(language: str, code: str) -> list:
    lang = language.lower()
    issues = []

    # Division by zero (direct)
    if re.search(r"/\s*0\b", code):
        issues.append({
            "line": 0,
            "message": "Possible division by zero.",
            "fix": "Check denominator before division.",
        })

    # Division by zero using variable
    zero_vars = re.findall(r"\b([a-zA-Z_]\w*)\s*=\s*0\s*;", code)
    for var in zero_vars:
        if re.search(r"/\s*" + re.escape(var) + r"\b", code):
            issues.append({
                "line": 0,
                "message": f"Possible division by zero using variable '{var}'.",
                "fix": "Check denominator before division.",
            })
            break

    # C / C++ checks
    if lang in ["c", "cpp", "c++"]:
        if re.search(r"for\s*\(.*<=\s*\w+", code):
            issues.append({
                "line": 0,
                "message": "Possible array index out of bounds.",
                "fix": "Use < instead of <= in loop condition.",
            })

        if "nums.size()" in code and "<=" in code:
            issues.append({
                "line": 0,
                "message": "Possible vector out of bounds access.",
                "fix": "Use i < nums.size() instead of i <= nums.size().",
            })

        # Detect declared variables
        declared_vars = set(
            re.findall(
                r"\b(?:int|float|double|char|string|bool|long|auto|vector<.*?>)\s+([a-zA-Z_]\w*)",
                code,
            )
        )

        # Extract cout statements safely
        cout_matches = re.findall(r"cout\s*<<(.+?);", code, re.DOTALL)

        ignored = {
            "cout",
            "cin",
            "endl",
            "true",
            "false",
        }

        for match in cout_matches:
            vars_found = re.findall(r"\b([a-zA-Z_]\w*)\b", match)

            for var in vars_found:
                if (
                    var not in declared_vars
                    and var not in ignored
                    and not var.isdigit()
                ):
                    issues.append({
                        "line": 0,
                        "message": f"Possible undeclared variable usage: '{var}'.",
                        "fix": f"Declare '{var}' before using it.",
                    })
                    return issues

    # JavaScript checks
    if lang == "javascript":
        if "response.json()" in code and "await response.json()" not in code:
            issues.append({
                "line": 0,
                "message": "response.json() is asynchronous and should be awaited.",
                "fix": "Use: const data = await response.json();",
            })

        if "i.age" in code:
            issues.append({
                "line": 0,
                "message": "Incorrect object access: i is an index, not object.",
                "fix": "Use users[i].age instead of i.age.",
            })

    # Java checks
    if lang == "java":
        if re.search(r"\bString\s+\w+\s*=\s*null\s*;", code) and ".length()" in code:
            issues.append({
                "line": 0,
                "message": "Possible NullPointerException.",
                "fix": "Check for null before calling methods.",
            })

        if re.search(r"<=\s*\w+\.length", code):
            issues.append({
                "line": 0,
                "message": "Possible array index out of bounds.",
                "fix": "Use < array.length instead of <= array.length.",
            })
        # PHP checks
    if lang == "php":
        if re.search(r"/\s*0\b", code):
            issues.append({
                "line": 0,
                "message": "Possible division by zero in PHP.",
                "fix": "Check denominator before division.",
            })

        if "while" in code and "$i++" not in code and "$i--" not in code:
            issues.append({
                "line": 0,
                "message": "Possible infinite loop in PHP.",
                "fix": "Update loop variable.",
            })

    # C# / .NET checks
    if lang in ["csharp", "c#", "dotnet"]:
        if re.search(r"/\s*0\b", code):
            issues.append({
                "line": 0,
                "message": "Possible division by zero in C#.",
                "fix": "Validate denominator before division.",
            })

        if "while" in code and "i++" not in code and "i--" not in code:
            issues.append({
                "line": 0,
                "message": "Possible infinite loop in C#.",
                "fix": "Update loop condition.",
            })

    # SQL Server checks
    if lang in ["sql", "sqlserver"]:
        if "delete from" in code.lower() and "where" not in code.lower():
            issues.append({
                "line": 0,
                "message": "DELETE without WHERE detected.",
                "fix": "Add WHERE clause before DELETE.",
            })

        if "drop table" in code.lower():
            issues.append({
                "line": 0,
                "message": "Dangerous DROP TABLE operation.",
                "fix": "Avoid dropping production tables.",
            })

    # HTML / Bootstrap checks
    if lang in ["html", "bootstrap"]:
        if "<html>" not in code.lower():
            issues.append({
                "line": 0,
                "message": "Missing HTML root tag.",
                "fix": "Add <html> tag.",
            })

        if "<body>" not in code.lower():
            issues.append({
                "line": 0,
                "message": "Missing body tag.",
                "fix": "Add <body> tag.",
            })

        if "container" not in code.lower() and "bootstrap" in lang:
            issues.append({
                "line": 0,
                "message": "Bootstrap container missing.",
                "fix": "Use container/container-fluid class.",
            })


    return issues

def get_language_errors(language: str, code: str) -> List[Dict[str, Any]]:
    lang = language.lower()

    if lang == "python":
        return get_python_syntax_errors(code)

    if lang == "javascript":
        return check_js_syntax(code)

    if lang == "java":
        return check_java_code(code)

    if lang == "c":
        compile_errors = check_c_code(code)
        if compile_errors:
            return compile_errors
        return check_static_errors(language, code)

    if lang in ["cpp", "c++"]:
        compile_errors = check_cpp_code(code)
        if compile_errors:
            return compile_errors
        return check_static_errors(language, code)
    if lang in ["php", "csharp", "c#", "dotnet", "sql", "sqlserver", "html", "bootstrap"]:
        return check_static_errors(language, code)
    return []

def fast_dl_status() -> Dict[str, Any]:
    return {
        "trainedExamples": 0,
        "similarExamplesFound": 0,
        "confidence": 0,
        "learnedSuggestions": [],
        "status": "Fast mode",
    }


def finalize_with_learning(language: str, code: str, result: Dict[str, Any]) -> Dict[str, Any]:
    try:
        dl_insights = build_dl_insights(language, code)
        result = merge_learned_suggestions(result, dl_insights)
        result["suggestions"] = clean_suggestions(result.get("suggestions", []))

        if len(code) < 5000:
            add_training_example(language, code, result)

    except Exception:
        result["dlInsights"] = fast_dl_status()

    return result


def inject_runtime_error(output: str, result: Dict[str, Any], language: str) -> Dict[str, Any]:
    current_errors = result.get("errors", [])

    if output.startswith("Error:"):
        message = output.replace("Error:", "").strip()

        if "infinite loop" in message.lower():
            fix = "Add increment/decrement or update loop condition."
        elif "division by zero" in message.lower() or "/ by zero" in message.lower():
            fix = "Check denominator before division."
        elif "nullpointer" in message.lower():
            fix = "Check for null before accessing object methods."
        elif "arrayindex" in message.lower() or "index" in message.lower():
            fix = "Check array/list index bounds before access."
        else:
            fix = f"Handle this {language} runtime issue safely."

        runtime_error = {
            "line": 0,
            "message": message,
            "fix": fix,
        }

        already_present = any(
            str(err.get("message", "")).strip().lower() == message.lower()
            for err in current_errors
        )

        if not already_present:
            current_errors.append(runtime_error)

    result["errors"] = current_errors
    result["fixedCode"] = generate_fixed_code(language, result.get("fixedCode", ""), current_errors)
    return result


def make_fast_result(language: str, code: str, output: str = "Cannot be determined safely.") -> Dict[str, Any]:
    return {
        "errors": [],
        "suggestions": [],
        "output": output,
        "fixedCode": generate_fixed_code(language, code, []),
        "score": {
            "readability": 90,
            "performance": 90,
            "maintainability": 90,
            "security": 85,
        },
        "dlInsights": fast_dl_status(),
    }


def ai_review(language: str, code: str) -> Dict[str, Any]:
    language = language.strip()
    lang = language.lower()

    syntax_or_compile_errors = get_language_errors(language, code)

    if syntax_or_compile_errors:
        result = {
            "errors": syntax_or_compile_errors,
            "suggestions": [
                err["fix"] for err in syntax_or_compile_errors
            ],
            "output": "Cannot be determined safely.",
            "fixedCode": generate_fixed_code(language, code, syntax_or_compile_errors),
            "score": {
                "readability": 40,
                "performance": 40,
                "maintainability": 40,
                "security": 50,
            },
            "dlInsights": fast_dl_status(),
        }

        # don't stop here for C/C++
        if lang not in ["c", "cpp", "c++"]:
            return result

        if len(code.strip()) < 200:
            if lang == "python":
                output = get_python_output(code)
            elif lang == "javascript":
                output = get_js_output(code)
            elif lang == "java":
                output = get_java_output(code)
            elif lang == "c":
                output = run_c(code)
            elif lang in ["cpp", "c++"]:
                output = run_cpp(code)
            else:
                output = "Cannot be determined safely."

            result = make_fast_result(language, code, output)
            result = inject_runtime_error(output, result, language)
            return result

    if lang == "javascript":
        static_errors = check_static_errors(language, code)
        output = get_js_output(code) if len(code) < 1500 else "Cannot be determined safely."

        if static_errors:
            result = {
                "errors": static_errors,
                "suggestions": [],
                "output": output,
                "fixedCode": generate_fixed_code(language, code, static_errors),
                "score": {
                    "readability": 70,
                    "performance": 55,
                    "maintainability": 60,
                    "security": 75,
                },
                "dlInsights": fast_dl_status(),
            }
            result = inject_runtime_error(output, result, language)
            return result

        if is_simple_js_code(code):
            result = make_fast_result(language, code, output)
            result = inject_runtime_error(output, result, language)
            return result

    if lang == "java":
        static_errors = check_static_errors(language, code)
        output = get_java_output(code) if len(code) < 2000 else "Cannot be determined safely."

        if static_errors:
            result = {
                "errors": static_errors,
                "suggestions": [],
                "output": output,
                "fixedCode": generate_fixed_code(language, code, static_errors),
                "score": {
                    "readability": 70,
                    "performance": 55,
                    "maintainability": 60,
                    "security": 75,
                },
                "dlInsights": fast_dl_status(),
            }
            result = inject_runtime_error(output, result, language)
            return result

        if is_simple_java_code(code):
            result = make_fast_result(language, code, output)
            result = inject_runtime_error(output, result, language)
            return result

    if lang == "c":
        static_errors = check_static_errors(language, code)
        output = run_c(code)

        if static_errors:
            result = {
                "errors": static_errors,
                "suggestions": [],
                "output": output,
                "fixedCode": generate_fixed_code(language, code, static_errors),
                "score": {
                    "readability": 70,
                    "performance": 55,
                    "maintainability": 60,
                    "security": 75,
                },
                "dlInsights": fast_dl_status(),
            }
            result = inject_runtime_error(output, result, language)
            return result

        if len(code.strip()) < 1200:
            result = make_fast_result(language, code, output)
            result = inject_runtime_error(output, result, language)
            return result

    if lang in ["cpp", "c++"]:
        static_errors = check_static_errors(language, code)
        output = run_cpp(code)

        if static_errors:
            result = {
                "errors": static_errors,
                "suggestions": [],
                "output": output,
                "fixedCode": generate_fixed_code(language, code, static_errors),
                "score": {
                    "readability": 70,
                    "performance": 55,
                    "maintainability": 60,
                    "security": 75,
                },
                "dlInsights": fast_dl_status(),
            }
            result = inject_runtime_error(output, result, language)
            return result

        if len(code.strip()) < 1200:
            result = make_fast_result(language, code, output)
            result = inject_runtime_error(output, result, language)
            return result

    if lang == "python":
        output = get_python_output(code)

        if is_simple_python_code(code):
            result = make_fast_result(language, code, output)
            result = inject_runtime_error(output, result, language)
            return result
        
    if lang in ["php", "csharp", "c#", "dotnet", "sql", "sqlserver", "html", "bootstrap"]:
        static_errors = check_static_errors(language, code)

    if static_errors:
        return {
            "errors": static_errors,
            "suggestions": [],
            "output": "Static analysis completed.",
            "fixedCode": generate_fixed_code(language, code, static_errors),
            "score": {
                "readability": 70,
                "performance": 70,
                "maintainability": 70,
                "security": 75,
            },
            "dlInsights": fast_dl_status(),
        }

    return make_fast_result(
        language,
        code,
        "Static analysis completed successfully."
    )

    prompt = build_prompt(language, code)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": get_model_options(language),
            },
            timeout=45,
        )

        response.raise_for_status()

        data = response.json()
        raw_text = data.get("response", "").strip()

        try:
            cleaned = extract_json(raw_text)
            parsed = json.loads(cleaned)
            result = normalize_review(parsed, code, language)
        except Exception:
            result = {
                "errors": [],
                "suggestions": ["Could not fully parse AI response. Showing basic analysis."],
                "output": "Cannot be determined safely.",
                "fixedCode": generate_fixed_code(language, code, []),
                "score": {
                    "readability": 70,
                    "performance": 70,
                    "maintainability": 70,
                    "security": 70,
                },
            }

        if lang == "python":
            result["output"] = get_python_output(code)
        elif lang == "javascript":
            result["output"] = get_js_output(code) if len(code) < 1500 else result.get("output", "Cannot be determined safely.")
        elif lang == "java":
            result["output"] = get_java_output(code) if len(code) < 2000 else result.get("output", "Cannot be determined safely.")
        elif lang == "c":
            result["output"] = run_c(code)
        elif lang in ["cpp", "c++"]:
            result["output"] = run_cpp(code)

        result = inject_runtime_error(result["output"], result, language)

        if not result["errors"]:
            result["fixedCode"] = code
            result["score"] = {
                "readability": max(result["score"]["readability"], 80),
                "performance": max(result["score"]["performance"], 80),
                "maintainability": max(result["score"]["maintainability"], 80),
                "security": max(result["score"]["security"], 75),
            }

        result["suggestions"] = clean_suggestions(result.get("suggestions", []))
        return finalize_with_learning(language, code, result)

    except requests.exceptions.Timeout:
        fallback = default_response(
            code,
            "AI review took too long. Showing available static/runtime analysis.",
            language,
        )

        if lang == "python":
            fallback["output"] = get_python_output(code)
        elif lang == "javascript":
            fallback["output"] = get_js_output(code) if len(code) < 1500 else "Cannot be determined safely."
        elif lang == "java":
            fallback["output"] = get_java_output(code) if len(code) < 2000 else "Cannot be determined safely."
        elif lang == "c":
            fallback["output"] = run_c(code)
        elif lang in ["cpp", "c++"]:
            fallback["output"] = run_cpp(code)

        fallback = inject_runtime_error(fallback["output"], fallback, language)
        return fallback

    except Exception as e:
        fallback = default_response(code, f"AI review failed: {str(e)}", language)

        if lang == "python":
            fallback["output"] = get_python_output(code)
        elif lang == "javascript":
            fallback["output"] = get_js_output(code) if len(code) < 1500 else "Cannot be determined safely."
        elif lang == "java":
            fallback["output"] = get_java_output(code) if len(code) < 2000 else "Cannot be determined safely."
        elif lang == "c":
            fallback["output"] = run_c(code)
        elif lang in ["cpp", "c++"]:
            fallback["output"] = run_cpp(code)

        fallback = inject_runtime_error(fallback["output"], fallback, language)
        return fallback