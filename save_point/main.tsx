// import 'bootstrap/dist/css/bootstrap.min.css';

// import React, { StrictMode, useEffect, useState } from "react";
// import { createRoot } from "react-dom/client";

// interface WebSocketData {
//     frame: string;
//     behavior: string;
//     cropped_images: string[];
//     fps: number;
// }

// interface ResNetData {
//     resnet_predictions: string;
// }


// const LiveFeed: React.FC = () => {
//     const [imageSrc, setImageSrc] = useState<string | null>(null);
//     const [croppedImages, setCroppedImages] = useState<string[]>([]);
//     const [fps, setFps] = useState<number | null>(null); // Store FPS

//     const [resnetResults, setResnetResults] = useState<string>("Waiting for detection...")
//     useEffect(() => {
//         const socket = new WebSocket("ws://localhost:8765");


//         socket.onmessage = (event: MessageEvent) => {
//             try {
//                 const data: WebSocketData = JSON.parse(event.data);
//                 setImageSrc(`data:image/jpeg;base64,${data.frame}`);
//                 setCroppedImages(data.cropped_images || []);
//                 setFps(data.fps); // Update FPS
//             } catch (error) {
//                 console.error("Error parsing WebSocket message:", error);
//             }
//         };

//         socket.onclose = () => console.log("WebSocket connection closed");

//         const resnetSocket = new WebSocket("ws://localhost:8766");

//         resnetSocket.onmessage = (event: MessageEvent) => {
//             try {
//                 const data = JSON.parse(event.data);
//                 setResnetResults(data.resnet_predictions);
//             } catch (error) {
//                 console.error("Error parsing ResNet WebSocket message:", error);
//             }
//         };

//         resnetSocket.onclose = () => console.log("ResNet WebSocket closed");

//         return () => {
//             resnetSocket.close();
//             socket.close();
//         };
//     }, []);

//     // Function for handling button click
//     const handleActivityLogClick = () => {
//         console.log("hi")
//         // lagyan mo ng function haha
//     };
    
//     return (
//         // <div className="container">
//         //     <button className="activity-log-btn" onClick={handleActivityLogClick}>
//         //         <img src="/activitylog.png" alt="Activity Log" className="activity-log-icon" />
//         //         <p className="activity-log-label">Activity Log</p> 
//         //     </button>
//         //     <h2 className="title">🚦 TRAFFIC SIGN DETECTION LIVE CAMERA FEED 🚦</h2>

//         //     {fps !== null && (
//         //         <p className="fps">🟢 FPS: {fps}</p>
//         //     )}

//         //     {imageSrc ? (
//         //         <img src={imageSrc} alt="Live Feed" className="live-feed" />
//         //     ) : (
//         //         <p className="connecting">Connecting to camera...</p>
//         //     )}

//         //     <h3 className="behavior">{resnetResults}</h3>

//         //     <div className="cropped-images">
//         //         {croppedImages.map((img, index) => (
//         //             <img key={index} src={`data:image/jpeg;base64,${img}`} alt={`Detected ${index}`} className="cropped-image" />
//         //         ))}
//         //     </div>
//         // </div>
//         // <div>
//         //     <nav className="nav nav-tabs justify-content-center bg-secondary text-white text-reset text-white">
//         //             <a className="nav-link active " href="#" onClick={handleActivityLogClick}>Inference</a>
//         //             <a className="nav-link" href="#Logs">Logs</a>
//         //     </nav>



//         // </div>
//         <div>
//             <ul className="nav nav-tabs" id="myTab" role="tablist">
//                 <li className="nav-item">
//                     <a className="nav-link active" id="home-tab" data-toggle="tab" href="#" role="tab" aria-controls="home" aria-selected="true">Home</a>
//                 </li>
//                 <li className="nav-item">
//                     <a className="nav-link" id="profile-tab" data-toggle="tab" href="Logs.tsx" role="tab" aria-controls="profile" aria-selected="false">Profile</a>
//                 </li>
//                 <li className="nav-item">
//                     <a className="nav-link" id="contact-tab" data-toggle="tab" href="#contact" role="tab" aria-controls="contact" aria-selected="false">Contact</a>
//                 </li>
//             </ul>
//             <div className="tab-content" id="myTabContent">
//                 <div className="tab-pane fade show active" id="home" role="tabpanel" aria-labelledby="home-tab">...</div>
//                 <div className="tab-pane fade" id="profile" role="tabpanel" aria-labelledby="profile-tab">a</div>
//                 <div className="tab-pane fade" id="contact" role="tabpanel" aria-labelledby="contact-tab">.2</div>
//             </div>
            
//         </div>
        

//     );
// };

// // Render the UI
// createRoot(document.getElementById("root")!).render(
//     <StrictMode>
//         <LiveFeed />
//     </StrictMode>
// );
