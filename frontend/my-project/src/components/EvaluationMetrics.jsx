import React, { useState, useEffect } from "react";
import axios from "axios";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar } from "recharts";
import { motion } from "framer-motion";

const EvaluationMetrics = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [rawApiData, setRawApiData] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                // Correctly fetch from the all-data endpoint
                const response = await axios.get("http://localhost:5000/api/all-data");
                if (!response.data) throw new Error("Empty response from server.");
                
                // Store raw data for inspection
                setRawApiData(JSON.stringify(response.data, null, 2));
                
                console.log("FULL API RESPONSE:", response.data);
                setData(response.data);
            } catch (err) {
                setError("Failed to fetch evaluation metrics data.");
                console.error("Error fetching data:", err);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    if (loading) return <p>Loading metrics...</p>;
    if (error) return <p style={{ color: "red" }}>{error}</p>;
    if (!data) return <p>No data available.</p>;

    // Debug what data we actually have
    console.log("Data structure:", Object.keys(data));
    
    // The API response format has changed, so we need to adjust how we extract data
    // Process Feature Importance data
    let featureImportanceData = [];
    try {
        if (data.graphData && data.graphData.feature_importance) {
            console.log("Feature Importance Exists:", data.graphData.feature_importance);
            if (data.graphData.feature_importance.data) {
                console.log("Feature Importance Data:", data.graphData.feature_importance.data);
        
                featureImportanceData = Object.keys(data.graphData.feature_importance.data).map((key) => ({
                    feature: key,
                    importance: parseFloat(data.graphData.feature_importance.data[key]) || 0
                }));
            } else {
                console.error("ERROR: graphData.feature_importance.data is missing!");
            }
        } else {
            console.error("ERROR: graphData.feature_importance is missing from response!");
        }
    } catch (err) {
        console.error("Error processing feature importance data:", err);
    }
    
    // Process Forecast Distribution data
let forecastDistributionStats = null;
try {
    if (data.graphData && data.graphData.forecast_distribution) {
        console.log("Forecast Distribution Exists:", data.graphData.forecast_distribution);
        
        if (data.graphData.forecast_distribution.data) {
            console.log("Forecast Distribution Data:", data.graphData.forecast_distribution.data);
            forecastDistributionStats = data.graphData.forecast_distribution.data;
        } else {
            console.error("ERROR: graphData.forecast_distribution.data is missing!");
        }
    } else {
        console.error("ERROR: graphData.forecast_distribution is missing from response!");
    }
} catch (err) {
    console.error("Error processing forecast distribution data:", err);
}

    
    // Process Trend data
let trendData = [];
try {
    if (data.graphData && data.graphData.trend) {
        console.log("Trend Exists:", data.graphData.trend);
        
        if (data.graphData.trend.data && Array.isArray(data.graphData.trend.data.timestamps)) {
            console.log("Trend Data:", data.graphData.trend.data);
            
            const { timestamps, rolling_mean } = data.graphData.trend.data;
            
            trendData = timestamps.map((timestamp, index) => ({
                timestamp,
                trend: rolling_mean[index] || 0,
                formattedTime: new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            }));
        } else {
            console.error("ERROR: graphData.trend.data is missing or invalid!");
        }
    } else {
        console.error("ERROR: graphData.trend is missing from response!");
    }
} catch (err) {
    console.error("Error processing trend data:", err);
}

    
    // Process Volatility data
let volatilityData = [];
try {
    if (data.graphData && data.graphData.volatility) {
        console.log("Volatility Exists:", data.graphData.volatility);
        
        if (data.graphData.volatility.data && Array.isArray(data.graphData.volatility.data.timestamps)) {
            console.log("Volatility Data:", data.graphData.volatility.data);
            
            const { timestamps, volatility } = data.graphData.volatility.data;
            
            volatilityData = timestamps.map((timestamp, index) => ({
                timestamp,
                volatility: volatility[index] || 0,
                formattedTime: new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            }));
        } else {
            console.error("ERROR: graphData.volatility.data is missing or invalid!");
        }
    } else {
        console.error("ERROR: graphData.volatility is missing from response!");
    }
} catch (err) {
    console.error("Error processing volatility data:", err);
}

    
    // Process Risk data 
let riskData = [];
try {
    if (data.graphData && data.graphData.risk) {
        console.log("Risk Exists:", data.graphData.risk);
        
        if (data.graphData.risk.data && Array.isArray(data.graphData.risk.data.timestamps)) {
            console.log("Risk Data:", data.graphData.risk.data);
            
            const { timestamps, risk_values, statuses } = data.graphData.risk.data;
            
            riskData = timestamps.map((timestamp, index) => ({
                timestamp,
                risk: risk_values[index] || 0,
                status: statuses[index] || "Unknown",
                formattedTime: new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            }));
        } else {
            console.error("ERROR: graphData.risk.data is missing or invalid!");
        }
    } else {
        console.error("ERROR: graphData.risk is missing from response!");
    }
} catch (err) {
    console.error("Error processing risk data:", err);
}

    
    // Process Vitals data
let vitalsData = [];
try {
    if (data.graphData && data.graphData.vitals) {
        console.log("Vitals Exists:", data.graphData.vitals);
        
        if (data.graphData.vitals.data && Array.isArray(data.graphData.vitals.data.timestamps)) {
            console.log("Vitals Data:", data.graphData.vitals.data);
            
            const { timestamps, body_temperature, heart_rate, spo2 } = data.graphData.vitals.data;
            
            vitalsData = timestamps.map((timestamp, index) => ({
                timestamp,
                temperature: body_temperature[index] || 0,
                heartRate: heart_rate[index] || 0,
                spo2: spo2[index] || 0,
                formattedTime: new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            }));
        } else {
            console.error("ERROR: graphData.vitals.data is missing or invalid!");
        }
    } else {
        console.error("ERROR: graphData.vitals is missing from response!");
    }
} catch (err) {
    console.error("Error processing vitals data:", err);
}


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
                boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)"
            }}
        >
            {children}
        </motion.div>
    );

    return (
        <div style={{ padding: "20px", backgroundColor: "white", color: "black" }}>
            <h2>📊 Evaluation Metrics</h2>

            <div style={{ marginBottom: "20px" }}>
                <details style={{ cursor: "pointer" }} open>
                    <summary style={{ 
                        padding: "10px", 
                        backgroundColor: "#e9ecef", 
                        borderRadius: "5px",
                        fontWeight: "bold"
                    }}>
                        API Data and Debug Information
                    </summary>
                    <div style={{ 
                        maxHeight: "400px", 
                        overflow: "auto", 
                        padding: "10px", 
                        backgroundColor: "#f1f3f5", 
                        border: "1px solid #ced4da",
                        borderRadius: "5px",
                        marginTop: "10px"
                    }}>
                        <h4>Available Data Types:</h4>
                        <ul>
                            {data && typeof data === 'object' && Object.keys(data).map(key => (
                                <li key={key}>{key}: {typeof data[key]}</li>
                            ))}
                        </ul>
                        <h4>Raw Data:</h4>
                        <pre>{rawApiData}</pre>
                    </div>
                </details>
            </div>

            <div style={{ 
                display: "grid", 
                gridTemplateColumns: "repeat(auto-fit, minmax(600px, 1fr))", 
                gap: "20px" 
            }}>
                {/* Feature Importance Graph */}
                <GraphContainer>
                    <h3>📌 Feature Importance</h3>
                    {featureImportanceData.length > 0 ? (
                        <BarChart 
                            width={600} 
                            height={400} 
                            data={featureImportanceData.sort((a, b) => b.importance - a.importance)} 
                            layout="vertical" 
                        >
                            <CartesianGrid stroke="lightgray" />
                            <XAxis type="number" domain={[0, 1]} stroke="black" />
                            <YAxis dataKey="feature" type="category" stroke="black" width={150} tick={{fontSize: 12}} />
                            <Tooltip formatter={(value) => value.toFixed(4)} />
                            <Legend />
                            <Bar dataKey="importance" fill="dodgerblue" animationDuration={1000} />
                        </BarChart>
                    ) : data?.feature_importance?.image ? (
                        <div>
                            <img 
                                src={`data:image/png;base64,${data.feature_importance.image}`} 
                                alt="Feature Importance" 
                                style={{maxWidth: '100%', maxHeight: '400px'}}
                            />
                        </div>
                    ) : (
                        <div style={{padding: "20px", textAlign: "center", backgroundColor: "#f8d7da", color: "#721c24", borderRadius: "5px"}}>
                            No feature importance data available.
                        </div>
                    )}
                </GraphContainer>
                
                {/* Forecast Distribution Stats */}
                <GraphContainer>
                    <h3>📊 Forecast Distribution Statistics</h3>
                    {forecastDistributionStats ? (
                        <div style={{padding: "20px"}}>
                            <table style={{width: "100%", borderCollapse: "collapse"}}>
                                <thead>
                                    <tr>
                                        <th style={{padding: "8px", textAlign: "left", borderBottom: "2px solid #ddd"}}>Statistic</th>
                                        <th style={{padding: "8px", textAlign: "right", borderBottom: "2px solid #ddd"}}>Value</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {Object.entries(forecastDistributionStats).map(([stat, value]) => (
                                        <tr key={stat}>
                                            <td style={{padding: "8px", textAlign: "left", borderBottom: "1px solid #ddd"}}>
                                                {stat.charAt(0).toUpperCase() + stat.slice(1)}
                                            </td>
                                            <td style={{padding: "8px", textAlign: "right", borderBottom: "1px solid #ddd"}}>
                                                {typeof value === 'number' ? value.toFixed(2) : value}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    ) : data?.forecast_distribution?.image ? (
                        <div>
                            <img 
                                src={`data:image/png;base64,${data.forecast_distribution.image}`} 
                                alt="Forecast Distribution" 
                                style={{maxWidth: '100%', maxHeight: '400px'}}
                            />
                        </div>
                    ) : (
                        <div style={{padding: "20px", textAlign: "center", backgroundColor: "#f8d7da", color: "#721c24", borderRadius: "5px"}}>
                            No forecast distribution data available.
                        </div>
                    )}
                </GraphContainer>
                
                {/* Trend (Rolling Mean) Chart */}
                <GraphContainer>
                    <h3>📈 Trend (Rolling Mean)</h3>
                    {trendData.length > 0 ? (
                        <LineChart 
                            width={600} 
                            height={300} 
                            data={trendData}
                            margin={{ top: 5, right: 30, left: 20, bottom: 25 }}
                        >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis 
                                dataKey="formattedTime"
                                tick={{ fontSize: 12, angle: -45, textAnchor: 'end' }}
                                height={60}
                                interval={Math.ceil(trendData.length / 10)} // Show around 10 tick marks
                            />
                            <YAxis 
                                domain={['auto', 'auto']}
                                label={{ value: 'Value', angle: -90, position: 'insideLeft' }}
                            />
                            <Tooltip 
                                formatter={(value) => [value.toFixed(2), "Rolling Mean"]}
                                labelFormatter={(label) => `Time: ${label}`}
                            />
                            <Legend verticalAlign="top" />
                            <Line 
                                type="monotone" 
                                dataKey="trend" 
                                stroke="#0066cc" 
                                strokeWidth={2}
                                name="Rolling Mean"
                                dot={false}
                                activeDot={{ r: 6 }}
                                animationDuration={1500}
                            />
                        </LineChart>
                    ) : data?.trend?.image ? (
                        <div>
                            <img 
                                src={`data:image/png;base64,${data.trend.image}`} 
                                alt="Trend Chart" 
                                style={{maxWidth: '100%', maxHeight: '400px'}}
                            />
                        </div>
                    ) : (
                        <div style={{padding: "20px", textAlign: "center", backgroundColor: "#f8d7da", color: "#721c24", borderRadius: "5px"}}>
                            No trend data available.
                        </div>
                    )}
                </GraphContainer>
                
                {/* Volatility Chart */}
                <GraphContainer>
                    <h3>📉 Volatility (Moving Standard Deviation)</h3>
                    {volatilityData.length > 0 ? (
                        <LineChart 
                            width={600} 
                            height={300} 
                            data={volatilityData}
                            margin={{ top: 5, right: 30, left: 20, bottom: 25 }}
                        >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis 
                                dataKey="formattedTime"
                                tick={{ fontSize: 12, angle: -45, textAnchor: 'end' }}
                                height={60}
                                interval={Math.ceil(volatilityData.length / 10)}
                            />
                            <YAxis 
                                domain={[0, 'auto']}
                                label={{ value: 'Std Deviation', angle: -90, position: 'insideLeft' }}
                            />
                            <Tooltip 
                                formatter={(value) => [value.toFixed(4), "Standard Deviation"]}
                                labelFormatter={(label) => `Time: ${label}`}
                            />
                            <Legend verticalAlign="top" />
                            <Line 
                                type="monotone" 
                                dataKey="volatility" 
                                stroke="#e63946" 
                                strokeWidth={2}
                                name="Volatility"
                                dot={false}
                                activeDot={{ r: 6 }}
                                animationDuration={1500}
                            />
                        </LineChart>
                    ) : data?.volatility?.image ? (
                        <div>
                            <img 
                                src={`data:image/png;base64,${data.volatility.image}`} 
                                alt="Volatility Chart" 
                                style={{maxWidth: '100%', maxHeight: '400px'}}
                            />
                        </div>
                    ) : (
                        <div style={{padding: "20px", textAlign: "center", backgroundColor: "#f8d7da", color: "#721c24", borderRadius: "5px"}}>
                            No volatility data available.
                        </div>
                    )}
                </GraphContainer>
            </div>

            <hr style={{ margin: "30px 0", border: "none", borderTop: "1px solid #dee2e6" }} />

            <div style={{ marginTop: "20px" }}>
                <h3>Data Format Instructions</h3>
                <p>The dashboard is now configured for the following API response format:</p>
                <pre style={{backgroundColor: "#f8f9fa", padding: "10px", borderRadius: "5px"}}>
{`{
  "feature_importance": {
    "image": "base64_encoded_image_string",
    "data": {
      "Feature 1": 0.85,
      "Feature 2": 0.75
    }
  },
  "forecast_distribution": {
    "image": "base64_encoded_image_string",
    "data": {
      "mean": 37.2,
      "median": 37.1,
      "std": 0.4
    }
  },
  "trend": {
    "image": "base64_encoded_image_string",
    "data": {
      "timestamps": ["2023-05-18T15:30:45.123456", ...],
      "rolling_mean": [45.6, 46.4, ...]
    }
  },
  "volatility": {
    "image": "base64_encoded_image_string",
    "data": {
      "timestamps": ["2023-05-18T15:30:45.123456", ...],
      "volatility": [0, 1.13, ...]
    }
  }
}`}
                </pre>
            </div>
        </div>
    );
};

export default EvaluationMetrics;
