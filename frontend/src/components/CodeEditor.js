import React from "react";
import Editor from "@monaco-editor/react";

let timeout; // debounce variable

function CodeEditor({ code, setCode, reviewCode }) {

  const handleChange = (value) => {
    setCode(value || "");

    // 🔥 DEBOUNCE LIVE REVIEW
    clearTimeout(timeout);

    timeout = setTimeout(() => {
      reviewCode(value);
    }, 800); // delay
  };

  return (
    <div className="editor">

      <h2>Code Editor</h2>

      <div style={{
        borderRadius: "8px",
        overflow: "hidden",
        border: "1px solid #ddd"
      }}>
        <Editor
          height="450px"
          defaultLanguage="javascript"
          value={code}
          onChange={handleChange}
          theme="vs-dark"
          options={{
            fontSize: 14,
            minimap: { enabled: false },
            automaticLayout: true,
            scrollBeyondLastLine: false
          }}
        />
      </div>

      <button
        onClick={() => reviewCode(code)}
        style={{
          marginTop: "20px",
          padding: "10px 20px",
          background: "#2563eb",
          color: "white",
          border: "none",
          borderRadius: "6px",
          cursor: "pointer",
          fontSize: "16px"
        }}
      >
        Review Code
      </button>

    </div>
  );
}

export default CodeEditor;