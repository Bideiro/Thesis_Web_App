// import React, { useEffect, useState } from "react";
// interface WebSocketData {
//     frame: string;
//     behavior: string;
//     cropped_images: string[];
//     fps: number; // Add FPS data
// }

// const LiveFeed: React.FC = () => {
//     const [imageSrc, setImageSrc] = useState<string | null>(null);
//     const [behavior, setBehavior] = useState<string>("Waiting for detection...");
//     const [croppedImages, setCroppedImages] = useState<string[]>([]);
//     const [fps, setFps] = useState<number | null>(null); // Store FPS

//     useEffect(() => {
//         const socket = new WebSocket(`ws://${window.location.hostname}:8765`);

//         socket.onmessage = (event: MessageEvent) => {
//             try {
//                 const data: WebSocketData = JSON.parse(event.data);
//                 setImageSrc(`data:image/jpeg;base64,${data.frame}`);
//                 setBehavior(data.behavior);
//                 setCroppedImages(data.cropped_images || []);
//                 setFps(data.fps); // Update FPS
//             } catch (error) {
//                 console.error("Error parsing WebSocket message:", error);
//             }
//         };

//         socket.onclose = () => console.log("WebSocket connection closed");

//         return () => socket.close();
//     }, []);

//     return (
//         <div className="p-4 text-center">
//             <h2 className="text-xl font-bold mb-4">Real Time Traffic Sign Detection - YOLO ResNetV250 Hybrid</h2>
            
//             {/* FPS Display */}
//             {fps !== null && <p className="text-green-500 font-semibold">FPS: {fps}</p>}

//             {imageSrc ? (
//                 <img src={imageSrc} alt="Live Feed" className="rounded-md shadow-md mx-auto" />
//             ) : (
//                 <p className="text-gray-500">Connecting to camera...</p>
//             )}
            
//             <h3 className="mt-4 text-lg font-semibold text-red-500">{behavior}</h3>

//             {/* Display cropped images */}
//             <div className="mt-4 grid grid-cols-2 gap-2">
//                 {croppedImages.map((img, index) => (
//                     <img key={index} src={`data:image/jpeg;base64,${img}`} alt={`Detected ${index}`} className="rounded-md shadow-md" />
//                 ))}
//             </div>
//         </div>
//     );
// };

// export default LiveFeed;



import React, { useEffect, useState } from "react";

interface WebSocketData {
    frame: string;
    behavior: string;
    cropped_images: string[];
    fps: number;
}

const LiveFeed: React.FC = () => {
    const [imageSrc, setImageSrc] = useState<string | null>(null);
    const [behavior, setBehavior] = useState<string>("Waiting for detection...");
    const [croppedImages, setCroppedImages] = useState<string[]>([]);
    const [fps, setFps] = useState<number | null>(null);

    useEffect(() => {
        const socket = new WebSocket(`ws://${window.location.hostname}:8765`);

        socket.onmessage = (event: MessageEvent) => {
            try {
                const data: WebSocketData = JSON.parse(event.data);
                setImageSrc(`data:image/jpeg;base64,${data.frame}`);
                setBehavior(data.behavior || "No detection");
                setCroppedImages(data.cropped_images || []);
                setFps(data.fps);
            } catch (error) {
                console.error("Error parsing WebSocket message:", error);
            }
        };

        socket.onclose = () => console.log("WebSocket connection closed");

        return () => socket.close();
    }, []);

    return (
        <div className="p-4 text-center">
            <h2 className="text-xl font-bold mb-4">Real-Time Traffic Sign Detection - YOLOv8 + ResNet50V2</h2>
            
            {fps !== null && <p className="text-green-500 font-semibold">FPS: {fps}</p>}
            
            {imageSrc ? (
                <img src={imageSrc} alt="Live Feed" className="rounded-md shadow-md mx-auto" />
            ) : (
                <p className="text-gray-500">Connecting to camera...</p>
            )}
            
            <h3 className="mt-4 text-lg font-semibold text-red-500">{behavior}</h3>

            <div className="mt-4 grid grid-cols-2 gap-2">
                {croppedImages.map((img, index) => (
                    <img key={index} src={`data:image/jpeg;base64,${img}`} alt={`Detected ${index}`} className="rounded-md shadow-md" />
                ))}
            </div>
        </div>
    );
};

export default LiveFeed;

