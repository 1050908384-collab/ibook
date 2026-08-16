import axios from 'axios';

// UI preview is intentionally local-only: no request is sent until a backend
// integration is explicitly enabled in the frontend environment.
export const isUiPreviewMode = import.meta.env.VITE_UI_PREVIEW === 'true';

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL
  || `${window.location.protocol}//${window.location.hostname}:8500/api/v1`;
const BASE_URL = configuredBaseUrl.replace(/\/$/, '');

const CLIENT_USER_ID_KEY = 'aiAgentTrainer:userId';

export function getClientUserId(): string {
  const existing = localStorage.getItem(CLIENT_USER_ID_KEY)?.trim();
  if (existing) return existing;
  const userId = `anon-${crypto.randomUUID()}`;
  localStorage.setItem(CLIENT_USER_ID_KEY, userId);
  return userId;
}

axios.defaults.headers.common['X-User-ID'] = getClientUserId();

axios.interceptors.request.use((config) => {
  config.headers.set('X-User-ID', getClientUserId());
  return config;
});

axios.interceptors.response.use(
  (response) => {
    if (import.meta.env.DEV) {
      console.debug('[API response]', response.config.method?.toUpperCase(), response.config.url, response.status);
    }
    return response;
  },
  (error) => {
    const response = error.response;
    console.error('[API request failed]', {
      method: error.config?.method?.toUpperCase(),
      url: error.config?.url,
      status: response?.status ?? 'network-error',
      detail: response?.data ?? error.message
    });
    return Promise.reject(error);
  }
);

export type ApiProfile = {
  profile_id?: string;
  session_id?: string;
  confirmed?: boolean;
  goal?: string;
  grade?: string;
  skill_tags?: string[];
  foundation_confirmed?: boolean;
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
  subtopic_id?: string;
  estimated_hours?: number;
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
  user_id?: string;
  user_profile_exists?: boolean;
  requires_full_profile?: boolean;
  target_settings?: Record<string, unknown>;
};

export type DiagnosisResponse = {
  status: 'needs_more_info' | 'profile_ready' | string;
  assistant_message: string;
  missing_fields: string[];
  followup_questions: string[];
  profile_draft: ApiProfile;
  confidence: number;
};

export type GoalBlock = {
  block_id: string;
  block_name: string;
  description?: string;
  difficulty?: string;
  estimated_hours?: number;
  subtopics: Array<{
    subtopic_id: string;
    subtopic_name: string;
    sequence_index?: number;
    prerequisites?: string[];
  }>;
};

export type GoalBlocksResponse = {
  goal: string;
  blocks: GoalBlock[];
};

export type FoundationResponse = {
  status: 'needs_more_info' | 'profile_ready' | string;
  profile?: ApiProfile | null;
  stage?: ApiStage | null;
  followup_questions?: string[];
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
  created_at?: string;
  updated_at?: string;
  goal?: string;
  grade?: string;
  skill_tags?: string[];
  knowledge_gaps?: Array<Record<string, unknown>>;
  daily_hours?: number;
  target_total_hours?: number;
  target_daily_hours?: number;
  learning_style?: string;
  version?: number;
  current_stage_num?: number;
  current_stage_name?: string;
  progress?: number;
  profile_summary?: Record<string, unknown>;
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

export type KnowledgeNetworkSubtopicStatus = {
  subtopic_id: string;
  block_id: string;
  subtopic_name?: string;
  sequence_index?: number;
  prerequisites?: string[];
  status: number;
  mastery_score: number;
  is_initial_foundation: boolean;
  attempt_count: number;
};

export type KnowledgeNetworkResponse = {
  user_id: string;
  total_subtopics: number;
  mastered_subtopics: number;
  blocks: Array<{
    block_id: string;
    mastered_count: number;
    subtopics: KnowledgeNetworkSubtopicStatus[];
  }>;
};

export type TargetSettingsResponse = {
  session_id: string;
  goal?: string;
  target_total_hours: number;
  target_daily_hours: number;
  target_progress?: number;
  adjusted_stages?: Array<{
    stage_id: string;
    estimated_hours: number;
  }>;
};

export type BlockSubtopicsResponse = {
  block_id: string;
  block_name: string;
  subtopics: Array<{
    subtopic_id: string;
    subtopic_name: string;
    sequence_index?: number;
    prerequisites?: string[];
  }>;
};

export type StageDurationResponse = {
  stage_id: string;
  session_id: string;
  stage_num?: number;
  estimated_hours: number;
};

export type DashboardResponse = {
  current_accuracy: number | null;
  average_accuracy: number | null;
  stage_history: ApiStage[];
  [key: string]: unknown;
};

export const apiClient = {
  getUserProfile: async () => {
    const userId = encodeURIComponent(getClientUserId());
    const response = await axios.get<ApiProfile & { user_id: string; default_daily_hours?: number }>(
      `${BASE_URL}/users/${userId}/profile`
    );
    return response.data;
  },

  updateUserProfile: async (data: {
    grade?: string;
    skill_tags?: string[];
    foundation_confirmed?: boolean;
    learning_style?: string;
    hardware?: string;
    default_daily_hours?: number;
    learning_rhythm?: string;
    mastered_subtopic_ids?: string[];
  }) => {
    const userId = encodeURIComponent(getClientUserId());
    const response = await axios.patch<ApiProfile & { user_id: string; default_daily_hours?: number }>(
      `${BASE_URL}/users/${userId}/profile`,
      data
    );
    return response.data;
  },

  getKnowledgeNetwork: async () => {
    const userId = encodeURIComponent(getClientUserId());
    const response = await axios.get<KnowledgeNetworkResponse>(
      `${BASE_URL}/users/${userId}/knowledge-network`
    );
    return response.data;
  },

  getBlockSubtopics: async (blockId: string) => {
    const response = await axios.get<BlockSubtopicsResponse>(
      `${BASE_URL}/knowledge-blocks/${encodeURIComponent(blockId)}/subtopics`
    );
    return response.data;
  },

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

  deleteSession: async (sessionId: string) => {
    const response = await axios.delete(`${BASE_URL}/sessions/${sessionId}`);
    return response.data;
  },

  deleteLastStage: async (sessionId: string) => {
    const response = await axios.delete(`${BASE_URL}/sessions/${sessionId}/stages/last`);
    return response.data as { session_id: string; deleted: boolean; session_deleted: boolean; details: Record<string, unknown> };
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

  getGoalBlocks: async (sessionId: string, goal: string) => {
    const response = await axios.post<GoalBlocksResponse>(
      `${BASE_URL}/sessions/${sessionId}/diagnosis/goal-blocks`,
      { goal }
    );
    return response.data;
  },

  submitFoundation: async (
    sessionId: string,
    data: { goal: string; mastered_subtopic_ids: string[]; daily_hours: number; learning_style: string; reuse_user_profile?: boolean }
  ) => {
    const response = await axios.post<FoundationResponse>(
      `${BASE_URL}/sessions/${sessionId}/diagnosis/foundation`,
      data
    );
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

  updateTargetSettings: async (
    sessionId: string,
    data: { target_total_hours?: number; target_daily_hours?: number }
  ) => {
    const response = await axios.patch<TargetSettingsResponse>(
      `${BASE_URL}/sessions/${sessionId}/target-settings`,
      data
    );
    return response.data;
  },

  getTargetSettings: async (sessionId: string) => {
    const response = await axios.get<TargetSettingsResponse>(
      `${BASE_URL}/sessions/${sessionId}/target-settings`
    );
    return response.data;
  },

  updateStageDuration: async (sessionId: string, stageId: string, estimatedHours: number) => {
    const response = await axios.patch<ApiStage>(
      `${BASE_URL}/sessions/${sessionId}/stages/${stageId}/duration`,
      { estimated_hours: estimatedHours }
    );
    return response.data;
  },

  getStageDuration: async (sessionId: string, stageId: string) => {
    const response = await axios.get<StageDurationResponse>(
      `${BASE_URL}/sessions/${sessionId}/stages/${stageId}/duration`
    );
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

  completeStage: async (sessionId: string, options?: { bypassAssessment?: boolean }) => {
    const params = new URLSearchParams();
    if (options?.bypassAssessment) {
      params.set('bypass_assessment', 'true');
    }
    const qs = params.toString() ? `?${params.toString()}` : '';
    const response = await axios.post(`${BASE_URL}/sessions/${sessionId}/stages/current/complete${qs}`);
    return response.data as {
      accuracy?: number;
      bypass_assessment?: boolean;
      phase?: string;
      updated_profile?: ApiProfile | null;
      state?: { stage_info?: ApiStage | null };
      stage_summary?: string;
      next_strategy?: string;
      can_plan_next_stage?: boolean;
      all_mastered?: boolean;
      session_status?: string;
      final_report?: Record<string, unknown>;
      available_actions?: string[];
    };
  },

  planNextStage: async (sessionId: string, adjustmentNotes = '') => {
    const response = await axios.post<PlanNextStageResponse>(`${BASE_URL}/sessions/${sessionId}/stages/next/plan`, {
      adjustment_notes: adjustmentNotes
    });
    return response.data;
  },

  getDashboard: async (sessionId: string) => {
    const response = await axios.get<DashboardResponse>(`${BASE_URL}/sessions/${sessionId}/dashboard`);
    return response.data;
  },

  // Compatibility for older, currently unmounted pages.
  getProgress: async (sessionId: string) => {
    const response = await axios.get(`${BASE_URL}/sessions/${sessionId}/dashboard`);
    return response.data;
  }
};
