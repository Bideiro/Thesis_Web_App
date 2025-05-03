import 'bootstrap/dist/css/bootstrap.min.css';
import React, { useEffect, useState } from "react";

interface WebSocketData {
    frame: string;
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
    const [resnetResults, setResnetResults] = useState<ResnetData[]>([]);

    // 0 = disabled, 1 = beep, 2 = voice
    const [soundEnabled, setSoundEnabled] = useState<number>(0);
    const [playingAudio, setPlayingAudio] = useState(false);
    const [currentAudio, setCurrentAudio] = useState<HTMLAudioElement | null>(null);

    let lastPlayed = 0; // this will track the last time a sound was played

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

                    const now = Date.now();
                    if (
                        soundEnabled !== 0 &&
                        parsed.ResNetResult.length > 0 &&
                        now - lastPlayed > 3000 // 3 seconds cooldown
                    ) {

                        const playResultsSequentially = async () => {
                            for (const result of parsed.ResNetResult) {
                                const classText = result.result;
                                const className = classText.split(":")[1]?.split("(")[0]?.trim();

                                if (className) {
                                    let audioFile = "";

                                    if (soundEnabled === 1) {
                                        audioFile = "/audio/beep.mp3";
                                    } else if (soundEnabled === 2) {
                                        audioFile = `/audio/${className}.mp3`;
                                    }

                                    const audio = new Audio(audioFile);
                                    setCurrentAudio(audio);
                                    await audio.play();

                                    lastPlayed = Date.now(); // Update timestamp
                                    await new Promise(res => setTimeout(res, 1000)); // wait 1 second before next
                                }
                            }
                        };

                        // Call the async function
                        playResultsSequentially();

                    }
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
    }, [soundEnabled]);

    return (
        <div className="container-fluid p-4">
            <h1 className="text-center mb-4">LIVE CAMERA FEED</h1>

            {/* Sound Control */}
            <div className='d-flex flex-row flex-wrap align-items-center gap-4 mb-4'>
                <div className="form-check">
                    <input
                        className="form-check-input"
                        type="radio"
                        name="soundOptions"
                        id="disableSound"
                        checked={soundEnabled === 0}
                        onChange={() => setSoundEnabled(0)}
                    />
                    <label className="form-check-label" htmlFor="disableSound">
                        Disable Sound
                    </label>
                </div>

                <div className="form-check">
                    <input
                        className="form-check-input"
                        type="radio"
                        name="soundOptions"
                        id="enableBeep"
                        checked={soundEnabled === 1}
                        onChange={() => setSoundEnabled(1)}
                    />
                    <label className="form-check-label" htmlFor="enableBeep">
                        Enable Beep
                    </label>
                </div>

                <div className="form-check">
                    <input
                        className="form-check-input"
                        type="radio"
                        name="soundOptions"
                        id="enableVoice"
                        checked={soundEnabled === 2}
                        onChange={() => setSoundEnabled(2)}
                    />
                    <label className="form-check-label" htmlFor="enableVoice">
                        Enable Voice
                    </label>
                </div>
                <p className="h6 text-left mb-0">🟢 FPS: {fps}</p>
            </div>

            {/* Live Camera Feed */}
            {imageSrc ? (
                <img
                    src={imageSrc}
                    alt="Live Feed"
                    className="img-fluid border border-primary border-5 rounded w-100"
                    style={{ maxHeight: "60vh", objectFit: "contain" }}
                />
            ) : (
                <p className="connecting text-center">Connecting to camera...</p>
            )}

            {/* ResNet Results */}
            <div className="border border-3 mt-4 p-3 bg-light rounded">
                <div className="row d-flex justify-content-center">
                    {resnetResults.map((resnet, index) => (
                        <div className="col-12 col-md-6 d-flex align-items-center mb-3" key={index}>
                            <div className="col-auto">
                                <img
                                    src={`data:image/jpeg;base64,${resnet.cropped_img}`}
                                    alt={`Detected ${index}`}
                                    className="img-fluid rounded"
                                    style={{ maxWidth: "150px", border: "2px solid orange" }}
                                />
                            </div>
                            <div className="col text-start ms-3">
                                <strong className="text-success">{resnet.result}</strong>
                            </div>
                            <div className="col-auto">
                                {resnet.class_img ? (
                                    <img
                                        src={`data:image/png;base64,${resnet.class_img}`}
                                        alt={`Reference ${index}`}
                                        className="img-fluid rounded"
                                        style={{ maxWidth: "150px", border: "3px solid red" }}
                                    />
                                ) : (
                                    <p className="text-muted">Waiting...</p>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default Inference;
