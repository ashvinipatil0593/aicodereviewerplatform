import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

const originalConsoleError = window.console.error;

window.console.error = (...args) => {
  if (
    typeof args[0] === "string" &&
    args[0].includes("ResizeObserver loop completed with undelivered notifications")
  ) {
    return;
  }
  originalConsoleError(...args);
};

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);