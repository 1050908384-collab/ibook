from pathlib import Path
from typing import List

class Skill:
    def __init__(self, name: str, content: str):
        self.name = name
        self.content = content

SKILLS_DIR = Path(__file__).parent

def load_skills(skill_names: List[str]) -> List[Skill]:
    skills = []
    for name in skill_names:
        skill_path = SKILLS_DIR / f"{name}.skill.md"
        if skill_path.exists():
            with open(skill_path, "r", encoding="utf-8") as f:
                skills.append(Skill(name, f.read()))
    return skills

def load_all_skills() -> List[Skill]:
    skills = []
    for skill_file in SKILLS_DIR.glob("*.skill.md"):
        name = skill_file.stem.replace(".skill", "")
        with open(skill_file, "r", encoding="utf-8") as f:
            skills.append(Skill(name, f.read()))
    return skills
