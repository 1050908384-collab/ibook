import { knowledgeGraphBlocks, type KnowledgeGraphBlock } from '../data/knowledgeGraph';

function normalize(value: string) {
  return value.trim().toLocaleLowerCase().replace(/[\s，。、；：:;,/\\()[\]{}（）「」“”‘’_+-]+/g, '');
}

function blockMatchesTag(block: KnowledgeGraphBlock, tag: string) {
  const normalizedTag = normalize(tag);
  if (!normalizedTag) return false;
  return [block.block_id, block.block_name, ...block.aliases].some((candidate) => {
    const normalizedCandidate = normalize(candidate);
    return normalizedTag === normalizedCandidate
      || normalizedTag.includes(normalizedCandidate)
      || normalizedCandidate.includes(normalizedTag);
  });
}

const FOUNDATION_STOP_TERMS = new Set(['学习', '基础', '理解', '掌握', '环境', '安装', '使用', '进行', '配置', '机制', '模型', '方法', '核心', '操作', '数据', '训练']);

function meaningfulTerms(value: string) {
  const terms = new Set<string>();
  value.match(/[a-z0-9]+|[\u4e00-\u9fff]{2,}/gi)?.forEach((part) => {
    if (/^[\u4e00-\u9fff]+$/.test(part)) {
      for (let index = 0; index < part.length - 1; index += 1) {
        const term = part.slice(index, index + 2);
        if (!FOUNDATION_STOP_TERMS.has(term)) terms.add(term);
      }
    } else if (part.length >= 2) {
      terms.add(part.toLocaleLowerCase());
    }
  });
  return [...terms];
}

export function foundationBlocksFromTags(tags: string[] = []) {
  return knowledgeGraphBlocks.filter((block) => tags.some((tag) => blockMatchesTag(block, tag)));
}

export function foundationNamesFromTags(tags: string[] = []) {
  return foundationBlocksFromTags(tags).map((block) => block.block_name);
}

export function getRelevantFoundationBlocks(goal: string) {
  const normalizedGoal = normalize(goal);
  if (!normalizedGoal) return knowledgeGraphBlocks;

  const matchedIds = new Set(
    knowledgeGraphBlocks
      .filter((block) => [block.block_id, block.block_name, ...block.aliases, ...block.subtopics.map((topic) => topic.subtopic_name)]
        .some((candidate) => {
          const normalizedCandidate = normalize(candidate);
          return normalizedCandidate.length >= 2
            && (normalizedGoal.includes(normalizedCandidate)
              || normalizedCandidate.includes(normalizedGoal)
              || meaningfulTerms(candidate).some((term) => normalizedGoal.includes(term)));
        }))
      .map((block) => block.block_id)
  );

  // Include direct prerequisites so the user can declare the full foundation path.
  let changed = true;
  while (changed) {
    changed = false;
    knowledgeGraphBlocks.forEach((block) => {
      if (!matchedIds.has(block.block_id)) return;
      block.prerequisites.forEach((prerequisiteId) => {
        if (!matchedIds.has(prerequisiteId)) {
          matchedIds.add(prerequisiteId);
          changed = true;
        }
      });
    });
  }

  return matchedIds.size
    ? knowledgeGraphBlocks.filter((block) => matchedIds.has(block.block_id))
    : knowledgeGraphBlocks;
}
