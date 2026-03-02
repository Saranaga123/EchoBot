const express = require('express');
const axios = require('axios');
const app = express();


app.use(express.json());

app.post('/ask', async (req, res) => {
    const { agent, question } = req.body;
  
    try {
      const response = await axios.post('http://localhost:3002/ask', {
        agent,
        question
      });
      res.json({ answer: response.data.answer });
    } catch (err) {
      console.error("Python server error:", err.message);
      res.status(500).json({ error: "Chatbot failed" });
    }
  });

app.listen(3001, () => {
    console.log("Express API running on http://localhost:3001");
});
