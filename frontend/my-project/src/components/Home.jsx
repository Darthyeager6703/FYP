import React from "react";
import member1 from "../assets/images/Siddharth DP.jpg"; // Adjust the path as needed
import member2 from "../assets/images/Shreya DP.jpg";
import member3 from "../assets/images/Harish DP.jpg";
import member4 from "../assets/images/Mithil DP.jpg";

const members = [
  { name: "Siddharth Mohanakrishnan", image: member1 },
  { name: "Shreya V", image: member2 },
  { name: "Harish Siddharth S", image: member3 },
  { name: "Mithil S", image: member4 },
];

const Home = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-b from-orange-500 to-yellow-300 text-white p-6">
      {/* Project Title */}
      <h1 className="text-4xl font-bold mb-6 text-center drop-shadow-lg">
        HEATSTROKE PREDICTION FROM PREDICTED HEATWAVE
      </h1>

      {/* Members Section */}
      <h2 className="text-2xl font-semibold mb-8">Project Members</h2>

      {/* Flexbox container for images */}
      <div className="flex flex-wrap justify-center gap-22">
        {members.map((member, index) => (
          <div key={index} className="flex flex-col items-center p-8">
            <img
              src={member.image} // Use the imported image directly
              alt={member.name}
              className="home-member-image transition-transform transform hover:scale-105" // Added hover effect
            />
            <p className="mt-3 text-lg font-medium">{member.name}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Home;
