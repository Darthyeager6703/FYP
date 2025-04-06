import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { motion } from 'framer-motion';
import './Hardware.css';

const Hardware = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [riskData, setRiskData] = useState([]);
  const [vitalsData, setVitalsData] = useState([]);

  const toggleConnection = async () => {
    const newConnectionState = !isConnected;
    setIsConnected(newConnectionState);
  
    try {
      await axios.post("http://localhost:5000/api/hardware-connection", { connected: newConnectionState });
      console.log("✅ Sent hardware connection status to backend:", newConnectionState);
    } catch (error) {
      console.error("❌ Error sending connection status:", error.message);
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get("http://localhost:5000/api/all-data");
        if (!response.data) throw new Error("Empty response from server.");
        
        console.log("FULL API RESPONSE:", response.data);
        setData(response.data);
        
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
        console.error("Error fetching data:", err);
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
    <div className="hardware-container">
      <h1 className="hardware-title">⚙️ Hardware ⚙️</h1>
      
      <div className="hardware-content">
        <div className="connection-status">
          <h2>Connection Status:</h2>
          
          <div className="toggle-container">
            <div 
              className={`toggle-button ${isConnected ? 'connected' : 'disconnected'}`}
              onClick={toggleConnection}
            >
              <div className="toggle-slider"></div>
            </div>
            <span className="toggle-label">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          
          <div className="status-details">
            {isConnected ? (
              <p>Hardware is currently connected and ready to use.</p>
            ) : (
              <p>Hardware is disconnected. Click the toggle to establish connection.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hardware;
