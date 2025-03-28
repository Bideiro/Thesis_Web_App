import 'bootstrap/dist/css/bootstrap.min.css';
import React, { useEffect, useState } from "react";

interface WebSocketData {
    frame: string;
    behavior: string;
    cropped_images: string[];
    fps: number;
}

interface ResNetData {
    resnet_predictions: string;
}

const Inference: React.FC = () => {
    const [imageSrc, setImageSrc] = useState<string | null>(null);
    const [croppedImages, setCroppedImages] = useState<string[]>([]);
    const [fps, setFps] = useState<number | null>(0);
    const [resnetResults, setResnetResults] = useState<string>("Waiting for detection...");

    useEffect(() => {
        // First Websocket
        const socket = new WebSocket("ws://localhost:8765");
        socket.onmessage = (event: MessageEvent) => {
            try {
                const data: WebSocketData = JSON.parse(event.data);
                setImageSrc(`data:image/jpeg;base64,${data.frame}`);
                setCroppedImages(data.cropped_images || []);
                setFps(data.fps); // Update FPS
            } catch (error) {
                console.error("Error parsing WebSocket message:", error);
            }
        };
        socket.onclose = () => console.log("WebSocket connection closed");

        // Second Websocket
        const resnetSocket = new WebSocket("ws://localhost:8766");
        resnetSocket.onmessage = (event: MessageEvent) => {
            try {
                const resnetpred: ResNetData = JSON.parse(event.data);
                setResnetResults(resnetpred.resnet_predictions);
            } catch (error) {
                console.error("Error parsing ResNet WebSocket message:", error);
            }
        };
        resnetSocket.onclose = () => console.log("ResNet WebSocket closed");

        // Proper Closing
        return () => {
            resnetSocket.close();
            socket.close();
        };
    }, []);

    return (
        <div className="container-fluid">
            <p className="display-1 text-center">LIVE CAMERA FEED</p>
            <p className="h6 text-left">🟢 FPS: {fps}</p>
    
            {/* Live Camera Feed */}
            {imageSrc ? (
                <img src={imageSrc} alt="Live Feed" className="img-fluid border border-primary border-5 rounded" />
            ) : (
                <p className="connecting">Connecting to camera...</p>
            )}
    
            {/* Cropped Images & ResNet Results (Responsive) */}
            <div className="border border-3 mt-3">
                <div className="row d-flex justify-content-center">
                    {croppedImages.map((img, index) => (
                        <div className="col-12 col-md-6 d-flex align-items-center border rounded p-2" key={index}>
                            {/* Image on the Left */}
                            <div className="col-auto">
                                <img 
                                    src={`data:image/jpeg;base64,${img}`} 
                                    alt={`Detected ${index}`} 
                                    className="img-fluid rounded"
                                    style={{ maxWidth: "150px", height: "auto" }}
                                />
                            </div>
    
                            {/* Text on the Right */}
                            <div className="col text-start ms-3">
                                <p className="fw-bold">{resnetResults[index]}</p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};
export default Inference;