const AUTH_TOKEN = "HFmD8JYKbVHWxSaL-VBVZCgxKdB5O5Ma";
const PINS = ["V0", "V1", "V2", "V3", "V4", "V5"]; // List of all pins

// Construct API URL with all pins
const API_URL = `https://blynk.cloud/external/api/get?token=${AUTH_TOKEN}&${PINS.map(pin => `${pin}`).join("&")}`;

async function fetchData() {
  try {
    const response = await fetch(API_URL);
    const data = await response.json();

    // Loop through all pins and log their values
    PINS.forEach(pin => {
      console.log(`Sensor Value from ${pin}:`, data[pin]); // Access data by key
    });

  } catch (error) {
    console.error("Error fetching data:", error.message);
  }
}

setInterval(fetchData, 1000); // Fetch new data every 5 seconds

