export type LearningStyle = 'theory' | 'practice' | 'mixed';

export type OnboardingFormData = {
  grade: string;
  skill_tags: string[];
  goal: string;
  daily_hours: string;
  learning_style: LearningStyle;
  hardware: string;
};

export type OnboardingField = keyof OnboardingFormData;

export const emptyOnboardingForm: OnboardingFormData = {
  grade: '',
  skill_tags: [],
  goal: '',
  daily_hours: '',
  learning_style: 'mixed',
  hardware: ''
};

export const learningStyleText: Record<LearningStyle, string> = {
  theory: '理论型',
  practice: '实操型',
  mixed: '混合型'
};

export function buildDiagnosticMessage(formData: OnboardingFormData) {
  const lines = [
    `我的身份背景：${formData.grade.trim()}`,
    `我的已有基础：${formData.skill_tags.join('、')}`,
    `我的学习目标：${formData.goal.trim()}`,
    `我每天可以学习：${Number(formData.daily_hours)} 小时`,
    `我的学习方式：${learningStyleText[formData.learning_style]}`,
  ];

  if (formData.hardware.trim()) {
    lines.push(`我的设备条件：${formData.hardware.trim()}`);
  }

  return lines.join('\n');
}

export function getFormCompleteness(formData: OnboardingFormData) {
  const checks = [
    Boolean(formData.grade.trim()),
    formData.skill_tags.length > 0,
    Boolean(formData.goal.trim()),
    isValidDailyHours(formData.daily_hours),
    Boolean(formData.learning_style),
  ];
  return checks.filter(Boolean).length / checks.length;
}

export function isOnboardingFormComplete(formData: OnboardingFormData) {
  return getFormCompleteness(formData) === 1;
}

export function isValidDailyHours(value: string) {
  const hours = Number(value);
  return Number.isFinite(hours) && hours >= 0.5 && hours <= 8;
}

export function formDataFromProfile(profile: {
  grade?: string;
  skill_tags?: string[];
  goal?: string;
  daily_hours?: number;
  learning_style?: string;
  hardware?: string;
}): OnboardingFormData {
  const style = profile.learning_style;
  return {
    grade: profile.grade || '',
    skill_tags: profile.skill_tags || [],
    goal: profile.goal || '',
    daily_hours: profile.daily_hours ? String(profile.daily_hours) : '',
    learning_style: style === 'theory' || style === 'practice' || style === 'mixed' ? style : 'mixed',
    hardware: profile.hardware || ''
  };
}
