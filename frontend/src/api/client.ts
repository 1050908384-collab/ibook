import axios from 'axios';

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8500/api/v1';
const BASE_URL = configuredBaseUrl.replace(/\/$/, '');

export type ApiProfile = {
  profile_id?: string;
  session_id?: string;
  confirmed?: boolean;
  goal?: string;
  grade?: string;
  skill_tags?: string[];
  daily_hours?: number;
  learning_style?: string;
  hardware?: string;
  knowledge_gaps?: Array<Record<string, unknown>>;
  learning_rhythm?: string;
  confidence_index?: number;
  confusion_markers?: string[];
  confidence?: number;
  version?: number;
};

export type ApiStage = Record<string, unknown> & {
  stage_id?: string;
  stage_num?: number;
  block_name?: string;
  status?: string;
};

export type ApiLearningPackage = Record<string, unknown> & {
  tutorial?: string | null;
  code_examples?: Array<Record<string, unknown>>;
  learning_resources?: Array<Record<string, unknown>>;
  questions?: Array<Record<string, unknown>>;
};

export type SessionSnapshot = {
  session_id: string;
  title?: string;
  status: string;
  phase: string;
  profile?: ApiProfile | null;
  profile_modified?: boolean;
  current_stage?: ApiStage | null;
  stage_info?: ApiStage | null;
  next_stage?: ApiStage | null;
  learning_package?: ApiLearningPackage | null;
  assessment_summary?: Record<string, unknown> | null;
  progress_summary?: Record<string, unknown> | null;
  recent_qa?: Array<Record<string, unknown>>;
  available_actions?: string[];
};

export type DiagnosisResponse = {
  status: 'needs_more_info' | 'profile_ready' | string;
  assistant_message: string;
  missing_fields: string[];
  followup_questions: string[];
  profile_draft: ApiProfile;
  confidence: number;
};

export type AssessmentAttempt = {
  attempt_id: string;
  session_id: string;
  stage_id?: string;
  status: string;
  total_questions: number;
  answered_count: number;
  questions: Array<Record<string, unknown>>;
  resumed?: boolean;
  accuracy?: number;
};

export type PlanNextStageResponse = {
  phase?: string;
  available_actions?: string[];
  updated_profile?: ApiProfile | null;
  next_stage?: ApiStage | null;
  can_plan_next_stage?: boolean;
};

export type SessionListItem = {
  session_id: string;
  title?: string;
  status?: string;
  phase?: string;
  updated_at?: string;
  goal?: string;
  grade?: string;
  skill_tags?: string[];
  knowledge_gaps?: Array<Record<string, unknown>>;
  daily_hours?: number;
  learning_style?: string;
  version?: number;
  current_stage_num?: number;
  current_stage_name?: string;
  progress?: number;
  profile_summary?: string;
};

export type SessionListResponse = {
  sessions: SessionListItem[];
};

export type DiagnosisHistoryResponse = {
  messages?: Array<{
    message_id?: string;
    role?: 'user' | 'assistant' | string;
    content?: string;
    created_at?: string;
  }>;
  total?: number;
};

export type QaHistoryResponse = {
  items?: Array<{
    qa_id?: string;
    question?: string;
    answer?: string;
    created_at?: string;
    review_summary?: {
      technical_passed?: boolean;
      quality_passed?: boolean;
      status?: string;
    };
  }>;
};

export const apiClient = {
  createSession: async () => {
    const response = await axios.post<SessionSnapshot>(`${BASE_URL}/sessions`);
    return response.data;
  },

  listSessions: async () => {
    const response = await axios.get<SessionListResponse>(`${BASE_URL}/sessions`);
    return response.data;
  },

  getSession: async (sessionId: string) => {
    const response = await axios.get<SessionSnapshot>(`${BASE_URL}/sessions/${sessionId}`);
    return response.data;
  },

  abandonSession: async (sessionId: string) => {
    const response = await axios.post(`${BASE_URL}/sessions/${sessionId}/abandon`);
    return response.data;
  },

  submitDiagnosis: async (sessionId: string, content: string) => {
    const response = await axios.post<DiagnosisResponse>(
      `${BASE_URL}/sessions/${sessionId}/diagnosis/messages`,
      { content, message_type: 'user' }
    );
    return response.data;
  },

  getDiagnosisMessages: async (sessionId: string) => {
    const response = await axios.get<DiagnosisHistoryResponse>(`${BASE_URL}/sessions/${sessionId}/diagnosis/messages`);
    return response.data;
  },

  getProfile: async (sessionId: string) => {
    const response = await axios.get<ApiProfile>(`${BASE_URL}/sessions/${sessionId}/profile`);
    return response.data;
  },

  confirmProfile: async (sessionId: string, notes = '') => {
    const response = await axios.post(`${BASE_URL}/sessions/${sessionId}/profile/confirm`, {
      confirm: true,
      notes
    });
    return response.data;
  },

  updateProfile: async (
    sessionId: string,
    data: { goal?: string; daily_hours?: number; learning_style?: string; hardware?: string }
  ) => {
    const response = await axios.patch(`${BASE_URL}/sessions/${sessionId}/profile`, data);
    return response.data;
  },

  getCurrentStage: async (sessionId: string) => {
    const response = await axios.get(`${BASE_URL}/sessions/${sessionId}/stages/current`);
    return response.data;
  },

  startStage: async (sessionId: string) => {
    const response = await axios.post(`${BASE_URL}/sessions/${sessionId}/stages/current/start`);
    return response.data;
  },

  getStagePackage: async (sessionId: string) => {
    const response = await axios.get(`${BASE_URL}/sessions/${sessionId}/stages/current/package`);
    return response.data;
  },

  askQuestion: async (sessionId: string, question: string) => {
    const response = await axios.post(`${BASE_URL}/sessions/${sessionId}/qa`, { question });
    return response.data;
  },

  getQaHistory: async (sessionId: string) => {
    const response = await axios.get<QaHistoryResponse>(`${BASE_URL}/sessions/${sessionId}/qa`);
    return response.data;
  },

  createAssessmentAttempt: async (sessionId: string) => {
    const response = await axios.post<AssessmentAttempt>(
      `${BASE_URL}/sessions/${sessionId}/stages/current/assessment-attempts`
    );
    return response.data;
  },

  getAssessmentAttempt: async (attemptId: string) => {
    const response = await axios.get<AssessmentAttempt>(`${BASE_URL}/assessment-attempts/${attemptId}`);
    return response.data;
  },

  saveAssessmentAnswer: async (
    attemptId: string,
    questionId: string,
    data: { user_answer: string; time_spent?: number }
  ) => {
    const response = await axios.put(
      `${BASE_URL}/assessment-attempts/${attemptId}/answers/${questionId}`,
      { user_answer: data.user_answer, time_spent: data.time_spent ?? 0 }
    );
    return response.data;
  },

  submitAssessmentAttempt: async (attemptId: string) => {
    const response = await axios.post(`${BASE_URL}/assessment-attempts/${attemptId}/submit`);
    return response.data;
  },

  completeStage: async (sessionId: string) => {
    const response = await axios.post(`${BASE_URL}/sessions/${sessionId}/stages/current/complete`);
    return response.data;
  },

  planNextStage: async (sessionId: string, adjustmentNotes = '') => {
    const response = await axios.post<PlanNextStageResponse>(`${BASE_URL}/sessions/${sessionId}/stages/next/plan`, {
      adjustment_notes: adjustmentNotes
    });
    return response.data;
  },

  getDashboard: async (sessionId: string) => {
    const response = await axios.get(`${BASE_URL}/sessions/${sessionId}/dashboard`);
    return response.data;
  },

  // Compatibility for older, currently unmounted pages.
  getProgress: async (sessionId: string) => {
    const response = await axios.get(`${BASE_URL}/sessions/${sessionId}/dashboard`);
    return response.data;
  }
};
