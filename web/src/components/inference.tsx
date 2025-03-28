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
    const [fps, setFps] = useState<number | null>(0); // Store FPS
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
                const data = JSON.parse(event.data);
                setResnetResults(data.resnet_predictions);
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
            <p className="display-1 text-center "> LIVE CAMERA FEED </p>
            <p className="h6 text-left">🟢 FPS: {fps}</p>

            {/* logic for null values */}
            {imageSrc ? (
                <img src={imageSrc} alt="Live Feed" className="img-fluid border rounded" />
            ) : (
                <p className="connecting">Connecting to camera...</p>
            )}

            <h3 className="behavior">{resnetResults}</h3>

            <div className="cropped-images">
                {croppedImages.map((img, index) => (
                    <img key={index} src={`data:image/jpeg;base64,${img}`} alt={`Detected ${index}`} className="border rounded" />
                ))}
            </div>
        </div>

    );
};
export default Inference;