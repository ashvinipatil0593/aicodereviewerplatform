import React, { useState } from "react";
import Editor from "@monaco-editor/react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  const [language, setLanguage] = useState("");
  const [code, setCode] = useState('print("hello")');
  const [review, setReview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showFullOutput, setShowFullOutput] = useState(false);

  const getScoreClass = (score) => {
    if (score >= 80) return "score-good";
    if (score >= 50) return "score-medium";
    return "score-bad";
  };

  const handleEditorWillMount = (monaco) => {
    monaco.languages.registerCompletionItemProvider("python", {
      provideCompletionItems: () => ({
        suggestions: [
          { label: "print", kind: monaco.languages.CompletionItemKind.Function, insertText: "print()" },
          { label: "for loop", kind: monaco.languages.CompletionItemKind.Snippet, insertText: "for i in range(10):\n    print(i)" },
          { label: "if condition", kind: monaco.languages.CompletionItemKind.Snippet, insertText: "if condition:\n    pass" },
          { label: "function", kind: monaco.languages.CompletionItemKind.Snippet, insertText: "def my_function():\n    pass" },
        ],
      }),
    });

    monaco.languages.registerCompletionItemProvider("javascript", {
      provideCompletionItems: () => ({
        suggestions: [
          { label: "console.log", kind: monaco.languages.CompletionItemKind.Function, insertText: "console.log();" },
          { label: "function", kind: monaco.languages.CompletionItemKind.Snippet, insertText: "function myFunc() {\n  \n}" },
        ],
      }),
    });
  };

  const handleReview = async () => {
    if (!language) {
      alert("⚠️ Please select a language before running audit.");
      return;
    }

    setLoading(true);
    setShowFullOutput(false);
    setReview(null);

    try {
      const response = await fetch(`${API_BASE}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ language, code }),
      });

      const text = await response.text();
      const data = JSON.parse(text);

      if (!response.ok) {
        throw new Error(data.detail || data.message || "Backend error");
      }

      setReview(data);
    } catch (error) {
      setReview({
        errors: [
          {
            line: 0,
            message: error.message || "Frontend failed to connect to backend.",
            fix: "Check backend terminal and API URL.",
          },
        ],
        suggestions: [],
        output: "Could not fetch output.",
        fixedCode: code,
        score: {
          readability: 0,
          performance: 0,
          maintainability: 0,
          security: 0,
        },
        dlInsights: {
          trainedExamples: 0,
          similarExamplesFound: 0,
          confidence: 0,
          learnedSuggestions: [],
          status: "Learning unavailable",
        },
      });
    } finally {
      setLoading(false);
    }
  };

  const outputText = review?.output || "No output available.";
  const fixedCodeText =
    review?.fixedCode && review.fixedCode.trim()
      ? review.fixedCode
      : "No fixes required. Code is already correct.";

  const shortOutput =
    outputText.length > 300 ? `${outputText.slice(0, 300)}...` : outputText;

  const overallScore = Math.round(
    ((review?.score?.readability ?? 0) +
      (review?.score?.performance ?? 0) +
      (review?.score?.maintainability ?? 0) +
      (review?.score?.security ?? 0)) /
      4
  );

  return (
    <div className="app">
      <aside className="sidebar">
        <h1 className="brand">⚡ DevAudit AI</h1>

        <select
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          className="select"
        >
          <option value="">Select Language</option>
          <option value="python">Python</option>
          <option value="javascript">JavaScript</option>
          <option value="java">Java</option>
          <option value="c">C</option>
          <option value="cpp">C++</option>
          <option value="php">PHP</option>
          <option value="csharp">C#</option>
          <option value="sql">SQL</option>
          <option value="html">HTML</option>
          <option value="bootstrap">Bootstrap</option>
        </select>

        <button className="btn" onClick={handleReview} disabled={loading}>
          {loading ? "Reviewing..." : "🚀 Run Audit"}
        </button>

        <label className="btn upload-btn">
          📁 Upload File
          <input
            type="file"
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (!file) return;

              const reader = new FileReader();
              reader.onload = (event) => {
                setCode(event.target?.result || "");
              };
              reader.readAsText(file);
            }}
          />
        </label>
      </aside>

      <main className="editor-panel">
        <div className="editor-header">
          <div className="editor-label">Editor</div>
          <div className="editor-tab">{language || "code"}.code</div>
        </div>

        <div className="editor-wrapper">
          <Editor
            beforeMount={handleEditorWillMount}
            language={language || "python"}
            theme="vs-dark"
            value={code}
            onChange={(value) => setCode(value || "")}
            height="100%"
            width="100%"
            options={{
              minimap: { enabled: false },
              fontSize: 16,
              automaticLayout: true,
              scrollBeyondLastLine: false,
              wordWrap: "on",
              lineNumbers: "on",
              tabSize: 2,
              insertSpaces: true,
              quickSuggestions: true,
              suggestOnTriggerCharacters: true,
              tabCompletion: "on",
            }}
          />
        </div>
      </main>

      <section className="review-panel">
        <h2 className="review-title">🧠 AI Reviewer</h2>

        {loading ? (
          <div className="card">
            <p>⏳ Reviewing your code...</p>
          </div>
        ) : !review ? (
          <div className="card">
            <p>
              Click <strong>Run Audit</strong> to review your code.
            </p>
          </div>
        ) : (
          <div className="result-layout">
            <div className="review-result-section">
              <div className="card">
                <h3>🤖 Deep Learning Insights</h3>

                <div className="score-line">
                  <span>Status</span>
                  <span>{review.dlInsights?.status || "Unavailable"}</span>
                </div>

                <div className="score-line">
                  <span>Trained Examples</span>
                  <span>{review.dlInsights?.trainedExamples ?? 0}</span>
                </div>

                <div className="score-line">
                  <span>Similar Examples Found</span>
                  <span>{review.dlInsights?.similarExamplesFound ?? 0}</span>
                </div>

                <div className="score-line">
                  <span>Learning Confidence</span>
                  <span>{review.dlInsights?.confidence ?? 0}/100</span>
                </div>
              </div>

              <div className="card">
                <h3>🐞 Errors</h3>

                {review.errors?.length ? (
                  review.errors.map((err, index) => (
                    <div key={index} className="item">
                      <strong>Line {err.line}:</strong> {err.message}
                      <div className="sub">{err.fix}</div>
                    </div>
                  ))
                ) : (
                  <p>No major errors found.</p>
                )}
              </div>

              <div className="card">
                <h3>💡 Suggestions</h3>

                {review.suggestions?.length ? (
                  <ul>
                    {review.suggestions.slice(0, 5).map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                ) : (
                  <p>No suggestions available.</p>
                )}
              </div>

              <div className="card">
                <h3>📊 Scores</h3>

                <div className="score-line">
                  <span>Readability</span>
                  <span className={getScoreClass(review.score?.readability ?? 0)}>
                    {review.score?.readability ?? 0}/100
                  </span>
                </div>

                <div className="score-line">
                  <span>Performance</span>
                  <span className={getScoreClass(review.score?.performance ?? 0)}>
                    {review.score?.performance ?? 0}/100
                  </span>
                </div>

                <div className="score-line">
                  <span>Maintainability</span>
                  <span className={getScoreClass(review.score?.maintainability ?? 0)}>
                    {review.score?.maintainability ?? 0}/100
                  </span>
                </div>

                <div className="score-line">
                  <span>Security</span>
                  <span className={getScoreClass(review.score?.security ?? 0)}>
                    {review.score?.security ?? 0}/100
                  </span>
                </div>

                <pre className="overall-score">
⭐ Overall Score: {overallScore}/100
                </pre>
              </div>
            </div>

            <div className="terminal-section">
              <div className="terminal-header">TERMINAL</div>

              <div className="terminal-block">
                <h3>📤 Predicted Output</h3>
                <pre>{showFullOutput ? outputText : shortOutput}</pre>

                {outputText.length > 300 && (
                  <button
                    className="btn small-btn"
                    onClick={() => setShowFullOutput(!showFullOutput)}
                  >
                    {showFullOutput ? "Show Less" : "Show More"}
                  </button>
                )}
              </div>

              <div className="terminal-block">
                <h3>🛠 Fixed Code</h3>
                <pre>{fixedCodeText}</pre>
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}