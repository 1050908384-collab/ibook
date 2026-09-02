export type StageResultActionMode =
  | 'review_only'
  | 'reinforce_and_start'
  | 'reinforce_and_plan'
  | 'start_next'
  | 'plan_next';

export function isHistoricalStageReview(
  displayedStageNum?: number | null,
  activeStageNum?: number | null
) {
  return Boolean(
    displayedStageNum
    && activeStageNum
    && displayedStageNum < activeStageNum
  );
}

export function resolveStageResultActionMode({
  allMastered,
  historicalReview,
  needsReinforcement,
  hasNextStage
}: {
  allMastered: boolean;
  historicalReview: boolean;
  needsReinforcement: boolean;
  hasNextStage: boolean;
}): StageResultActionMode {
  if (allMastered || historicalReview) return 'review_only';
  if (needsReinforcement) {
    return hasNextStage ? 'reinforce_and_start' : 'reinforce_and_plan';
  }
  return hasNextStage ? 'start_next' : 'plan_next';
}