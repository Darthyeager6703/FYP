import React, { useState, useEffect } from "react";
import "./InputForm.css"; // Make sure this file exists for styles

const InputForm = () => {
  const [formData, setFormData] = useState({
    name: "",
    age: "",
    gender: "",
    atmTemp: "",
    bodyTemp: "",
    heartRate: "",
    diastolicBP: "",
    systolicBP: "",
    humidity: "",
  });

  const [showModal, setShowModal] = useState(false);
  const [isVisible, setIsVisible] = useState(false); // Track visibility

  useEffect(() => {
    setTimeout(() => setIsVisible(true), 10); // Small delay to trigger animation
  }, []);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setShowModal(true);
  };

  return (
    <div className={`input-form-container ${isVisible ? "fade-in" : ""}`}>
      <h2 className="form-title">🔥 Input Form 🔥</h2>
      <form className="form-content" onSubmit={handleSubmit}>
        {["Name", "Age", "Gender", "Atmospheric Temp", "Body Temp", "Heart Rate", "Diastolic BP", "Systolic BP", "Relative Humidity outside"].map((label, index) => {
          const name = label.toLowerCase().replace(/\s+/g, "");
          return (
            <div key={index} className="form-group">
              <label className="form-label">{label}:</label>
              <input
                type="text"
                name={name}
                value={formData[name]}
                onChange={handleChange}
                required
                className="form-input"
              />
            </div>
          );
        })}
        <button type="submit" className="form-submit">Submit</button>
      </form>

      {/* Modal Confirmation */}
      {showModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h2>Are you sure?</h2>
            <div className="modal-buttons">
              <button onClick={() => setShowModal(false)} className="confirm-btn">Yes</button>
              <button onClick={() => setShowModal(false)} className="cancel-btn">No</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default InputForm;
