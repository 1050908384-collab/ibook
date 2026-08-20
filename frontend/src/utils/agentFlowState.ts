export const agentIds = ['agent1', 'agent3', 'agent4', 'agent6'] as const;

export type AgentId = typeof agentIds[number];
export type AgentRunState = 'pending' | 'running' | 'completed' | 'failed';
export type AgentStatusMap = Record<AgentId, AgentRunState>;

export function createAgentStatusMap(completed: AgentId[] = []): AgentStatusMap {
  const completedIds = new Set(completed);
  return Object.fromEntries(
    agentIds.map((agentId) => [agentId, completedIds.has(agentId) ? 'completed' : 'pending'])
  ) as AgentStatusMap;
}

export function startAgents(
  current: AgentStatusMap,
  runningAgents: AgentId[],
  reset = false
): AgentStatusMap {
  const runningIds = new Set(runningAgents);
  const base = reset ? createAgentStatusMap() : current;
  return Object.fromEntries(
    agentIds.map((agentId) => [
      agentId,
      runningIds.has(agentId)
        ? 'running'
        : base[agentId] === 'running'
          ? 'pending'
          : base[agentId]
    ])
  ) as AgentStatusMap;
}

export function finishAgents(
  current: AgentStatusMap,
  finishedAgents: AgentId[],
  succeeded: boolean
): AgentStatusMap {
  const finishedIds = new Set(finishedAgents);
  return Object.fromEntries(
    agentIds.map((agentId) => [
      agentId,
      finishedIds.has(agentId) ? (succeeded ? 'completed' : 'failed') : current[agentId]
    ])
  ) as AgentStatusMap;
}

export function restoreAgentStatuses(input: {
  hasProfile: boolean;
  hasLearningPackage: boolean;
  hasStageResult?: boolean;
  phase?: string;
}): AgentStatusMap {
  const completed: AgentId[] = [];
  if (input.hasProfile) completed.push('agent1');
  if (input.hasLearningPackage) completed.push('agent3', 'agent4');
  const agent6Completed = input.hasStageResult
    ?? ['stage_result', 'next_stage_ready', 'completed'].includes(input.phase || '');
  if (agent6Completed) completed.push('agent6');
  return createAgentStatusMap(completed);
}
