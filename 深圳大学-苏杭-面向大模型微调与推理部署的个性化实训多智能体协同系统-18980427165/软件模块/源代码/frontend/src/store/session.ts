import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../api/client';

interface SessionStore {
  sessionId: string | null;
  profile: any | null;
  currentStage: any | null;
  learningPackage: any | null;
  progress: any | null;
  loading: boolean;
}

export const useSessionStore = () => {
  const [store, setStore] = useState<SessionStore>({
    sessionId: null,
    profile: null,
    currentStage: null,
    learningPackage: null,
    progress: null,
    loading: false
  });

  useEffect(() => {
    const savedSessionId = localStorage.getItem('sessionId');
    if (savedSessionId) {
      setStore(prev => ({ ...prev, sessionId: savedSessionId }));
    }
  }, []);

  const createSession = useCallback(async () => {
    setStore(prev => ({ ...prev, loading: true }));
    try {
      const response = await apiClient.createSession();
      setStore(prev => ({ ...prev, sessionId: response.session_id, loading: false }));
      localStorage.setItem('sessionId', response.session_id);
      return response.session_id;
    } catch (error) {
      setStore(prev => ({ ...prev, loading: false }));
      throw error;
    }
  }, []);

  const fetchProfile = useCallback(async (sessionId: string) => {
    setStore(prev => ({ ...prev, loading: true }));
    try {
      const profile = await apiClient.getProfile(sessionId);
      setStore(prev => ({ ...prev, profile, loading: false }));
      return profile;
    } catch (error) {
      setStore(prev => ({ ...prev, loading: false }));
      throw error;
    }
  }, []);

  const fetchProgress = useCallback(async (sessionId: string) => {
    setStore(prev => ({ ...prev, loading: true }));
    try {
      const progress = await apiClient.getProgress(sessionId);
      setStore(prev => ({ ...prev, progress, loading: false }));
      return progress;
    } catch (error) {
      setStore(prev => ({ ...prev, loading: false }));
      throw error;
    }
  }, []);

  const startStage = useCallback(async (sessionId: string) => {
    setStore(prev => ({ ...prev, loading: true }));
    try {
      await apiClient.startStage(sessionId);
      const response = await apiClient.getStagePackage(sessionId);
      setStore(prev => ({ ...prev, learningPackage: response.package, loading: false }));
      return response;
    } catch (error) {
      setStore(prev => ({ ...prev, loading: false }));
      throw error;
    }
  }, []);

  const clearSession = useCallback(() => {
    setStore({
      sessionId: null,
      profile: null,
      currentStage: null,
      learningPackage: null,
      progress: null,
      loading: false
    });
    localStorage.removeItem('sessionId');
  }, []);

  return {
    ...store,
    createSession,
    fetchProfile,
    fetchProgress,
    startStage,
    clearSession
  };
};
