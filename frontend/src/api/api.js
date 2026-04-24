import axios from "axios";

const API_BASE = "http://localhost:8000";

export const trainModel = async () => {
  return axios.post(`${API_BASE}/train`);
};

export const evaluateModel = async () => {
  return axios.get(`${API_BASE}/evaluate`);
};

export const predictImage = async (formData) => {
  return axios.post(`${API_BASE}/predict`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
};
