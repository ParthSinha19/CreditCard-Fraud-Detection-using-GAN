import React, { useState } from 'react';
import axios from 'axios';

const GenerateGraph = () => {
  const [cardNumber, setCardNumber] = useState("");
  const [graphUrl, setGraphUrl] = useState("");

  const handleGenerateGraph = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:8000/generate", { card_number: cardNumber }, { responseType: 'blob' });
      const blob = new Blob([response.data], { type: 'image/png' });
      const url = URL.createObjectURL(blob);
      setGraphUrl(url);
    } catch (error) {
      console.error("Error generating graph:", error);
    }
  };

  return (
    <div>
      <h1>Confidence Thresholding</h1>
      <input
        type="text"
        placeholder="Enter Credit Card Number"
        value={cardNumber}
        onChange={(e) => setCardNumber(e.target.value)}
      />
      <button onClick={handleGenerateGraph}>Generate Graph</button>
      {graphUrl && (
        <div>
          <h3>Generated Confidence Graph:</h3>
          <img src={graphUrl} alt="Confidence Graph" />
        </div>
      )}
    </div>
  );
};

export default GenerateGraph;
