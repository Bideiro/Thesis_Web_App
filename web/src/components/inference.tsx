import 'bootstrap/dist/css/bootstrap.min.css';
import React, { useEffect, useState } from "react";

interface WebSocketData {
    frame: string;
    behavior: string;
    cropped_images: string[];
    fps: number;
}

interface ResnetData {
    result: string;
    class_img: string;
    cropped_img: string;
}

const Inference: React.FC = () => {
    const [imageSrc, setImageSrc] = useState<string | null>(null);
    const [fps, setFps] = useState<number | null>(0);

    const [croppedImages, setCroppedImages] = useState<string[]>([]);
    const [class_img, setclass_img] = useState<string | null>(null);
    const [resnetResults, setResnetResults] = useState<ResnetData[]>([]);

    // Voice
    const [soundEnabled, setSoundEnabled] = useState<number | null>(0);

    useEffect(() => {
        const socket = new WebSocket(`ws://${window.location.hostname}:8765`);
        socket.onmessage = (event: MessageEvent) => {
            try {
                const data: WebSocketData = JSON.parse(event.data);
                setImageSrc(`data:image/jpeg;base64,${data.frame}`);
                setFps(data.fps);
            } catch (error) {
                console.error("Error parsing WebSocket message:", error);
            }
        };
        socket.onclose = () => console.log("WebSocket connection closed");

        const resnetSocket = new WebSocket(`ws://${window.location.hostname}:8766`);
        resnetSocket.onmessage = (event: MessageEvent) => {
            try {
                const parsed = JSON.parse(event.data);
                if (parsed.ResNetResult && Array.isArray(parsed.ResNetResult)) {
                    setResnetResults(parsed.ResNetResult);
                }
            } catch (error) {
                console.error("Error parsing ResNet WebSocket message:", error);
            }
        };

        resnetSocket.onclose = () => console.log("ResNet WebSocket closed");

        return () => {
            resnetSocket.close();
            socket.close();
        };
    }, []);


    return (
        <div className="container-fluid">
            <p className="display-1 text-center">LIVE CAMERA FEED</p>
            <div className='d-flex flex-row flex-column-sm align-items-center gap-2'>
                <input
                    className="form-check-input"
                    type="radio"
                    name="toggleDisable"
                    id="disableSound"
                //   checked={!soundEnabled}
                //   onChange={() => {
                //     setSoundEnabled(false);
                //     stopCurrentAudio();
                //   }}
                />
                <label className="form-check-label" htmlFor="disableSound">
                    Disable Sound
                </label>

                <input
                    className="form-check-input"
                    type="radio"
                    name="toggleBeep"
                    id="Enable Beep"
                // checked={!soundEnabled}
                // onChange={() => {
                //     setSoundEnabled(false);
                //     stopCurrentAudio();
                // }}
                />
                <label className="form-check-label" htmlFor="disableSound">
                    Enable Beep
                </label>

                    <input
                        className="form-check-input"
                        type="radio"
                        name="toggleVoice"
                        id="enableVoice"
                        // checked={soundEnabled}
                        // onChange={() => setSoundEnabled(true)}
                    />
                    <label className="form-check-label" htmlFor="enableSound">
                        Enable Voice
                    </label>
                    <p className="h6 text-left">🟢 FPS: {fps}</p>
                </div>


                {/* Live Camera Feed */}
                {imageSrc ? (
                    <img src={imageSrc} alt="Live Feed" className="img-fluid border border-primary border-5 rounded" />
                ) : (
                    <p className="connecting">Connecting to camera...</p>
                )}

                {/* Cropped Images & ResNet Results (Responsive) */}
                <div className="border border-3 mt-3">
                    <div className="row d-flex justify-content-center">
                        {resnetResults.map((data, index) => {
                            const resnet = data;

                            return (
                                <div className="col-12 col-md-6 d-flex align-items-center rounded p-2" key={index}>
                                    {/* 🟡 Yellow box: Cropped image */}
                                    <div className="col-auto">
                                        <img
                                            src={`data:image/jpeg;base64,${resnet.cropped_img}`}
                                            alt={`Detected ${index}`}
                                            className="img-fluid rounded"
                                        />
                                    </div>

                                    {/* 🟢 Class Name */}
                                    <div
                                        className="col text-start ms-3"
                                        style={{
                                            marginRight: "20px",
                                            fontWeight: "bold",
                                            fontSize: "18px",
                                            color: "green",
                                        }}
                                    >
                                        {resnet ? resnet.result : "Detecting..."}
                                    </div>

                                    {/* 🔴 Reference Image (class_img) */}
                                    <div className="col-auto pe-3">
                                        {resnet?.class_img ? (
                                            <img
                                                src={`data:image/png;base64,${resnet.class_img}`}
                                                alt={`Reference ${index}`}
                                                className="img-fluid rounded"
                                                style={{ maxWidth: "150px", height: "auto", border: "3px solid red" }}
                                            />
                                        ) : (
                                            <p style={{ color: "gray" }}>Waiting...</p>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            </div>
            );
};

            export default Inference;
