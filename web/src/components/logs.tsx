import 'bootstrap/dist/css/bootstrap.min.css';
import React, { useEffect, useState } from "react";

interface logs {
    logged_image: string[];
    results: string[];
}

const Logs: React.FC = () => {
    const [results, setresults] = useState<string[]>([]);
    const [logged_image, setlogged_image] = useState<string[]>([]);

    useEffect(() => {
        // Third Websocket
        const logs_data_WS = new WebSocket("ws://localhost:8767");
        logs_data_WS.onmessage = (event: MessageEvent) => {
            try {
                const data: logs = JSON.parse(event.data);
                setresults(data.results);
                setlogged_image(data.logged_image);

            } catch (error) {
                console.error("Error parsing ResNet WebSocket message:", error);
            }
        };

        logs_data_WS.onclose = () => console.log("ResNet WebSocket closed");
        return () => {
            logs_data_WS.close();
        };
    }, []);

    return (
        <div className="container-fluid">
            <p className="display-1 text-center">Log List</p>
            <p className="lead text-center">
                A list that contains all detected signs, up to 10 signs, updates every 15 seconds.
            </p>

            <div className="container text-center">
                <div className="row d-flex flex-column align-items-center">
                    {logged_image.map((img, index) => (
                        <div className="col-8 d-flex align-items-center border rounded p-2 mb-3" key={index}>
                            {/* Image on the left */}
                            <div className="col-auto">
                                <img 
                                    src={`data:image/jpeg;base64,${img}`} 
                                    alt={`Logged Image ${index}`} 
                                    className="img-fluid rounded"
                                    style={{ maxWidth: "150px", height: "auto" }} // Controls image size
                                />
                            </div>

                            {/* Text on the right */}
                            <div className="col text-start ms-3">
                                <p className="fw-bold">Logged at:</p>
                                <p>{results[index] ?? "No timestamp available"}</p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );

    // return (
    //     <div className="container-fluid">
    //         <p className="display-1 text-center" >Log List</p>
    //         <p className='lead text-center'> A list that contains all detected signs, upto 10 signs updates every 15 seconds.</p>
    //         <div className="container text-center">
    //             <div className='row row-cols-2'>
    //                 {logged_image.map((img, index) => (

    //                     <div>
    //                         <div className='col' key={index}>
    //                             <img src={`data:image/jpeg;base64,${img}`} alt={`Logged Image ${index}`} />
    //                             <p>Logged at: {results[index]}</p>
    //                         </div>
    //                         {/* <div className='col' key={index}>
    //                             <p>Logged at: {results[index]}</p>
    //                         </div> */}
    //                     </div>
    //                     // <div className='col' key={index}>
    //                     //     <img src={`data:image/jpeg;base64,${img}`} alt={`Logged Image ${index}`} />
    //                     // </div>
    //                 ))}
    //             </div>
    //         </div>
    //     </div>
    // )
}

export default Logs;
