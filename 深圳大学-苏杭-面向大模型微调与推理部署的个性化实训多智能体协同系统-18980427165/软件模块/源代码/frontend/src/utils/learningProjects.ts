export type ProjectTimelineItem = {
  sessionId: string;
  goal: string;
  stageName?: string;
  createdAt?: number;
  updatedAt: number;
};

export function sortProjectsForDashboard<T extends ProjectTimelineItem>(
  projects: T[],
  _currentSessionId: string | null
) {
  return [...projects].sort((a, b) => {
    // 固定排序：最新创建的永远排最上面，不随当前会话切换跳动。
    // createdAt 缺失时回退到 updatedAt，最后仍有 updatedAt 作为 tie-breaker。
    const createdA = a.createdAt ?? a.updatedAt;
    const createdB = b.createdAt ?? b.updatedAt;
    const createdDifference = createdB - createdA;
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
