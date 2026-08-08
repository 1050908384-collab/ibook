export type ProjectTimelineItem = {
  sessionId: string;
  goal: string;
  stageName?: string;
  createdAt?: number;
  updatedAt: number;
};

export function sortProjectsForDashboard<T extends ProjectTimelineItem>(
  projects: T[],
  currentSessionId: string | null
) {
  return [...projects].sort((a, b) => {
    const aIsCurrent = a.sessionId === currentSessionId;
    const bIsCurrent = b.sessionId === currentSessionId;
    if (aIsCurrent !== bIsCurrent) return aIsCurrent ? -1 : 1;
    const createdDifference = (b.createdAt ?? b.updatedAt) - (a.createdAt ?? a.updatedAt);
    return createdDifference || b.updatedAt - a.updatedAt;
  });
}

export function filterProjectsForDashboard<T extends ProjectTimelineItem>(projects: T[], query: string) {
  const normalizedQuery = query.trim().toLocaleLowerCase('zh-CN');
  if (!normalizedQuery) return projects;
  return projects.filter((project) => `${project.goal} ${project.stageName || ''}`
    .toLocaleLowerCase('zh-CN')
    .includes(normalizedQuery));
}

export function formatProjectCreatedDate(timestamp?: number) {
  if (!timestamp || !Number.isFinite(timestamp)) return '日期未知';
  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit'
  }).format(new Date(timestamp));
}
