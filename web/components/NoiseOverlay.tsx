"use client";

import { useEffect, useState } from "react";

export default function NoiseOverlay() {
    const [noiseUrl, setNoiseUrl] = useState<string>("");

    useEffect(() => {
        const canvas = document.createElement("canvas");
        canvas.width = 200;
        canvas.height = 200;
        const ctx = canvas.getContext("2d");

        if (!ctx) return;

        const generateNoise = () => {
            const imageData = ctx.createImageData(200, 200);
            const data = imageData.data;

            for (let i = 0; i < data.length; i += 4) {
                const value = Math.floor(Math.random() * 255);
                data[i] = value;
                data[i + 1] = value;
                data[i + 2] = value;
                data[i + 3] = 255;
            }

            ctx.putImageData(imageData, 0, 0);
            setNoiseUrl(canvas.toDataURL("image/png"));
        };

        const animate = () => {
            generateNoise();
        };

        generateNoise();

        const intervalId = setInterval(animate, 200);

        return () => {
            clearInterval(intervalId);
        };
    }, []);

    if (!noiseUrl) return null;

    return (
        <div
            className="fixed inset-0 pointer-events-none z-[9999]"
            style={{
                backgroundImage: `url(${noiseUrl})`,
                backgroundRepeat: "repeat",
                opacity: 0.05,
            }}
            aria-hidden="true"
        />
    );
}
