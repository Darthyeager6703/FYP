import React, { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { motion } from 'framer-motion';
import axios from 'axios';
import "./WeatherTable.css"; // Import the CSS file

const WeatherTable = () => {
  const [patientData, setPatientData] = useState(null); // Store single risk entry
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [riskData, setRiskData] = useState([]);
  const [vitalsData, setVitalsData] = useState([]);

  useEffect(() => {
    // Fetch risk assessment from API
    const fetchData = async () => {
      try {
        const response = await axios.get("http://localhost:5000/api/all-data");
        if (!response.data) throw new Error("Empty response from server.");
        
        console.log("FULL API RESPONSE:", response.data);
        setData(response.data);
        
        // Extract risk assessment object
        const riskAssessment = response.data.riskAssessment || null;

        if (!riskAssessment) {
          console.error("No risk assessment data found");
          return;
        }

        // Extract vitals from response
        const vitals = riskAssessment.vitals || {};

        // Map vitals correctly from virtual pins
        const mappedVitals = {
          "Atmospheric Temperature (°C)": vitals.V0 ?? "N/A",
          "Relative Humidity (%)": vitals.V1 ?? "N/A",
          "Body Temperature (°C)": vitals.V2 ?? "N/A",
          "SpO2 (%)": vitals.V3 ?? "N/A",
          "Heart Rate (bpm)": vitals.V4 ?? "N/A",
        };

        // Set extracted data
        setPatientData({
          risk: `${riskAssessment["Risk (%)"].toFixed(2)}%`,
          status: riskAssessment["Status"] ?? "N/A",
          timestamp: new Date(riskAssessment["timestamp"]).toLocaleString(),
          vitals: mappedVitals,
        });
        
        // Process Risk data
        if (response.data.graphData?.risk?.data && Array.isArray(response.data.graphData.risk.data.timestamps)) {
          const { timestamps, risk_values, statuses } = response.data.graphData.risk.data;
          
          const processedRiskData = timestamps.map((timestamp, index) => ({
            timestamp,
            risk: risk_values[index] || 0,
            status: statuses[index] || "Unknown",
            formattedTime: new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
          }));
          
          setRiskData(processedRiskData);
        }
        
        // Process Vitals data
        if (response.data.graphData?.vitals?.data && Array.isArray(response.data.graphData.vitals.data.timestamps)) {
          const { timestamps, body_temperature, heart_rate, spo2 } = response.data.graphData.vitals.data;
          
          const processedVitalsData = timestamps.map((timestamp, index) => ({
            timestamp,
            temperature: body_temperature[index] || 0,
            heartRate: heart_rate[index] || 0,
            spo2: spo2[index] || 0,
            formattedTime: new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
          }));
          
          setVitalsData(processedVitalsData);
        }
      } catch (err) {
        setError("Failed to fetch data.");
        console.error("Error loading data:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Unified GraphContainer component
  const GraphContainer = ({ children }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="graph-container"
      style={{ 
        backgroundColor: "#f8f9fa", 
        padding: "15px", 
        borderRadius: "10px",
        boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
        margin: "20px 0"
      }}
    >
      {children}
    </motion.div>
  );

  return (
    <div>
      {/* Separator and spacing */}
      <div className="separator">
        <hr />
        <p className="separator-text">Heatstroke Risk Assessment 🩺</p>
        <hr />
      </div>

      {/* Patient Risk Card */}
      {patientData ? (
        <div className="patient-container">
          <div className="patient-card">
            <h3>Real-Time Risk Assessment</h3>
            <table className="vitals-table">
              <tbody>
                {Object.entries(patientData.vitals).map(([label, value]) => (
                  <tr key={label}>
                    <td className="vital-label">{label}</td>
                    <td className="vital-value">{value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="risk-text">Heatstroke Risk: <strong>{patientData.risk}</strong></p>
            <p className="status-text">Status: <strong>{patientData.status}</strong></p>
            <p className="timestamp">Last Updated: {patientData.timestamp}</p>
          </div>
        </div>
      ) : (
        <p>Loading risk assessment...</p>
      )}

      {/* Graphs Section - Below the risk output */}
      <div style={{ marginTop: "30px" }}>
        <h2>Historical Monitoring Data</h2>
        {loading && <p>Loading graph data...</p>}
        {error && <p style={{ color: "red" }}>{error}</p>}
        
        <div style={{ 
          display: "flex", 
          flexDirection: "row",
          flexWrap: "nowrap",
          gap: "20px",
          overflowX: "auto" // Allow horizontal scrolling if needed
        }}>
          {/* Risk Assessment Chart */}
          <GraphContainer>
            <h3>⚠️ Risk Assessment History</h3>
            {riskData.length > 0 ? (
              <LineChart 
                width={600} 
                height={300} 
                data={riskData}
                margin={{ top: 5, right: 30, left: 20, bottom: 25 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="formattedTime"
                  tick={{ fontSize: 12, angle: -45, textAnchor: 'end' }}
                  height={60}
                  interval={Math.ceil(riskData.length / 10)}
                />
                <YAxis 
                  domain={[0, 100]}
                  label={{ value: 'Risk Score', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip 
                  formatter={(value) => [value.toFixed(1), "Risk Score"]}
                  labelFormatter={(label) => `Time: ${label}`}
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div style={{backgroundColor: 'white', padding: '10px', border: '1px solid #ccc'}}>
                          <p>{`Time: ${label}`}</p>
                          <p>{`Risk Score: ${payload[0].value.toFixed(1)}`}</p>
                          <p>{`Status: ${payload[0].payload.status}`}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend verticalAlign="top" />
                <Line 
                  type="monotone" 
                  dataKey="risk" 
                  stroke="#ff9800" 
                  strokeWidth={2}
                  name="Risk Score"
                  dot={false}
                  activeDot={{ r: 6 }}
                  animationDuration={1500}
                />
              </LineChart>
            ) : data?.risk?.image ? (
              <div>
                <img 
                  src={`data:image/png;base64,${data.risk.image}`} 
                  alt="Risk Assessment" 
                  style={{maxWidth: '100%', maxHeight: '400px'}}
                />
              </div>
            ) : (
              <div style={{padding: "20px", textAlign: "center", backgroundColor: "#f8d7da", color: "#721c24", borderRadius: "5px"}}>
                No risk assessment history available.
              </div>
            )}
          </GraphContainer>
          
          {/* Vitals Chart */}
          <GraphContainer>
            <h3>❤️ Vital Signs History</h3>
            {vitalsData.length > 0 ? (
              <LineChart 
                width={600} 
                height={300} 
                data={vitalsData}
                margin={{ top: 5, right: 30, left: 20, bottom: 25 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="formattedTime"
                  tick={{ fontSize: 12, angle: -45, textAnchor: 'end' }}
                  height={60}
                  interval={Math.ceil(vitalsData.length / 10)}
                />
                <YAxis 
                  yAxisId="temp"
                  orientation="left"
                  domain={[35, 40]}
                  label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft' }}
                />
                <YAxis 
                  yAxisId="hr"
                  orientation="right"
                  domain={[40, 120]}
                  label={{ value: 'Heart Rate (bpm)', angle: 90, position: 'insideRight' }}
                />
                <Tooltip 
                  formatter={(value, name) => {
                    if (name === 'temperature') return [`${value.toFixed(1)}°C`, 'Body Temperature'];
                    if (name === 'heartRate') return [`${value} bpm`, 'Heart Rate'];
                    if (name === 'spo2') return [`${value}%`, 'SpO2'];
                    return [value, name];
                  }}
                  labelFormatter={(label) => `Time: ${label}`}
                />
                <Legend verticalAlign="top" />
                <Line 
                  type="monotone" 
                  dataKey="temperature" 
                  stroke="#e63946" 
                  strokeWidth={2}
                  name="Body Temperature"
                  yAxisId="temp"
                  dot={false}
                  activeDot={{ r: 6 }}
                />
                <Line 
                  type="monotone" 
                  dataKey="heartRate" 
                  stroke="#0066cc" 
                  strokeWidth={2}
                  name="Heart Rate"
                  yAxisId="hr"
                  dot={false}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            ) : data?.vitals?.image ? (
              <div>
                <img 
                  src={`data:image/png;base64,${data.vitals.image}`} 
                  alt="Vital Signs" 
                  style={{maxWidth: '100%', maxHeight: '400px'}}
                />
              </div>
            ) : (
              <div style={{padding: "20px", textAlign: "center", backgroundColor: "#f8d7da", color: "#721c24", borderRadius: "5px"}}>
                No vital signs history available.
              </div>
            )}
          </GraphContainer>
        </div>
      </div>
    </div>
  );
};

export default WeatherTable;
