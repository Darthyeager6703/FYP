const express = require("express");
const fs = require("fs");
const cors = require("cors");
const axios = require("axios");

const app = express();
app.use(cors()); // Allow frontend access
app.use(express.json()); // Parse JSON request bodies

// Hardware connection status
let isHardwareConnected = false;

const flaskBaseUrl = "http://localhost:5001"; // Ensure Flask runs on this

const AUTH_TOKEN = "HFmD8JYKbVHWxSaL-VBVZCgxKdB5O5Ma";
const PINS = ["V0", "V1", "V2", "V3", "V4", "V5"]; // Blynk Pins
const BLYNK_API_URL = `https://blynk.cloud/external/api/get?token=${AUTH_TOKEN}&${PINS.map(pin => `${pin}`).join("&")}`;

// ✅ Fetch Predictions
app.get("/api/predictions", (req, res) => {
    fs.readFile("predictions.json", "utf8", (err, data) => {
        if (err) {
            console.error("❌ Error reading predictions.json:", err);
            res.status(500).json({ error: "Failed to load predictions" });
        } else {
            res.json(JSON.parse(data));
        }
    });
});

// ✅ Fetch All Data (Includes Graphs, Risk Assessment, and Blynk Data)
app.get("/api/all-data", async (req, res) => {
    try {
        // Define Flask endpoints
        const endpoints = {
            graphData: "/graph_data",            
            riskAssessment: "/risk_assessment",  
        };

        // Fetch Flask data & Blynk data concurrently
        const responses = await Promise.all([
            ...Object.entries(endpoints).map(async ([key, path]) => {
                try {
                    const response = await axios.get(`${flaskBaseUrl}${path}`);
                    return { [key]: response.data };
                } catch (error) {
                    console.error(`❌ Error fetching ${path}:`, error.message);
                    return { [key]: { error: `Failed to fetch ${key}` } };
                }
            }),
            fetchBlynkData() // Fetch Blynk sensor data
        ]);

        // Merge responses
        const mergedResponse = responses.reduce((acc, obj) => ({ ...acc, ...obj }), {});

        res.json(mergedResponse);
    } catch (error) {
        console.error("❌ Error in all-data API:", error.message);
        res.status(500).json({ error: "Failed to fetch combined data" });
    }
});

// ✅ Fetch Blynk Data (Sensor Readings) - Skips 0 Values
async function fetchBlynkData() {
    try {
        const response = await axios.get(BLYNK_API_URL);
        let blynkData = response.data;
        
        let skippedKeys = [];
        for (let key in blynkData) {
            if (blynkData[key] === 0) {
                console.warn(`⚠️ Skipping ${key} as value is 0`);
                skippedKeys.push(key);
                delete blynkData[key]; // Remove 0 values from the response
            }
        }

        if (skippedKeys.length > 0) {
            console.warn(`⚠️ Skipped values: ${skippedKeys.join(", ")}`);
        }

        return { blynkData };
    } catch (error) {
        console.error("❌ Error fetching Blynk data:", error.message);
        return { blynkData: { error: "Failed to fetch Blynk data" } };
    }
}


// ✅ New endpoint to update hardware connection status
app.post("/api/hardware-connection", (req, res) => {
    const { connected } = req.body;
    isHardwareConnected = connected;
    console.log(`🔌 Hardware connection status: ${connected ? 'Connected' : 'Disconnected'}`);
    res.json({ success: true, status: connected });
});

async function sendBlynkDataToFlask() {
    // Only send data if hardware is connected
    if (!isHardwareConnected) {
        console.log("❌ Hardware disconnected - Not sending Blynk data");
        return;
    }

    try {
        const response = await axios.get(BLYNK_API_URL);
        const blynkData = response.data;

        // Check if any value is 0, and skip sending the data if true
        const hasZeroValue = Object.values(blynkData).includes(0);

        if (hasZeroValue) {
            const skippedKeys = Object.keys(blynkData).filter(key => blynkData[key] === 0);
            console.warn(`⚠️ Skipping values due to 0 values: ${skippedKeys.join(", ")}`);
            return; // Skip sending the data if any value is 0
        }

        // Send data to Flask server if no 0 values
        await axios.post("http://localhost:5001/blynk_data", blynkData);
        console.log("✅ Sent Blynk data to Flask server:", blynkData);
    } catch (error) {
        console.error("❌ Error sending Blynk data to Flask:", error.message);
    }
}


// ✅ Fetch and send Blynk data every 1 second
setInterval(sendBlynkDataToFlask, 1000);

// ✅ Keep Express server running on port 5000
app.listen(5000, () => console.log("🚀 Server running on port 5000"));
