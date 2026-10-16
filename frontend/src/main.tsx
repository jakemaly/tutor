// React and ReactDOM create the component tree inside index.html's root element.
import React from "react";
import ReactDOM from "react-dom/client";
// App is the chat screen; CSS is imported here so its styles apply across components.
import App from "./App";
import "./styles.css";

// StrictMode surfaces common development mistakes without adding anything to the UI.
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
