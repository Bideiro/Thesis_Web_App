// import React, { StrictMode } from "react";
// import { createRoot } from "react-dom/client";
// import App from "./app.tsx";

// createRoot(document.getElementById("root")!).render(
//   <StrictMode>
//     <App />
//   </StrictMode>
// );


import React, { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import LiveFeed from "./components/LiveFeed";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <LiveFeed />
    </div>
  </StrictMode>
);
