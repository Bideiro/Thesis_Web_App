import 'bootstrap/dist/css/bootstrap.min.css';
import React from "react";


function Logs() {
    return (
        <div style={{ textAlign: "center", padding: "50px" }}>
            <h1>Welcome to My Simple React Page 🚀</h1>
            <p>This is a basic React setup.</p>
            <button onClick={() => alert("Hello, React!")}>Click Me</button>
        </div>
    );
}

export default Logs;
