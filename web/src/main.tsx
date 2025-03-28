import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js';

import React, { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import Logs from "./components/logs";
import Inference from "./components/inference";

const Main = () => {
    return (
        <div >
            <ul className="nav nav-pills nav-justified" id="myTab" role="tablist">
                <li className="nav-item" role="presentation">
                    <button
                        className="nav-link active"
                        id="inference-tab"
                        data-bs-toggle="tab"
                        data-bs-target="#inference"
                        type="button"
                        aria-selected="true">
                        Inference
                    </button>
                </li>
                <li className="nav-item" role="presentation">
                    <button
                        className="nav-link"
                        id="logs-tab"
                        data-bs-toggle="tab"
                        data-bs-target="#logs"
                        type="button"
                        aria-selected="false"
                    >
                        Logs
                    </button>
                </li>
            </ul>

            <div className="tab-content" id="myTabContent">
                <div
                    className="tab-pane fade show active"
                    id="inference"
                    role="tabpanel"
                >
                    <Inference />
                </div>
                <div
                    className="tab-pane fade"
                    id="logs"
                    role="tabpanel"
                >
                    <Logs />
                </div>
            </div>
        </div>

    );
};

// Render the UI
createRoot(document.getElementById("root")!).render(
    <StrictMode>
        <Main />
    </StrictMode>
);
