import axios from 'axios';

const BASE_URL = 'http://localhost:8000/api';

export const apiClient = {
  chat: async (data: { message: string; session_id?: string }) => {
    const response = await axios.post(`${BASE_URL}/chat/`, data);
    return response.data;
  },
  
  handleQuestion: async (data: { session_id: string; message: string }) => {
    const response = await axios.post(`${BASE_URL}/chat/question`, data);
    return response.data;
  },
  
  createSession: async () => {
    const response = await axios.post(`${BASE_URL}/session/`);
    return response.data;
  },
  
  getSession: async (sessionId: string) => {
    const response = await axios.get(`${BASE_URL}/session/${sessionId}`);
    return response.data;
  },
  
  endSession: async (sessionId: string) => {
    const response = await axios.put(`${BASE_URL}/session/${sessionId}/end`);
    return response.data;
  },
  
  startStage: async (data: { session_id: string }) => {
    const response = await axios.post(`${BASE_URL}/stage/start`, data);
    return response.data;
  },
  
  submitAnswer: async (data: { session_id: string; question_id: string; user_answer: string }) => {
    const response = await axios.post(`${BASE_URL}/stage/submit-answer`, data);
    return response.data;
  },
  
  completeStage: async (sessionId: string) => {
    const response = await axios.post(`${BASE_URL}/stage/complete/${sessionId}`);
    return response.data;
  },
  
  getProfile: async (sessionId: string) => {
    const response = await axios.get(`${BASE_URL}/visualize/profile/${sessionId}`);
    return response.data;
  },
  
  getStage: async (sessionId: string) => {
    const response = await axios.get(`${BASE_URL}/visualize/stage/${sessionId}`);
    return response.data;
  },
  
  getProgress: async (sessionId: string) => {
    const response = await axios.get(`${BASE_URL}/visualize/progress/${sessionId}`);
    return response.data;
  }
};
