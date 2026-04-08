from functools import lru_cache
from pathlib import Path

SKILL_PATHS = {
    "project": [
        './skills'
    ]
}

# Skill 发现

def discover_project_skills(cwd: str):
    skills = []
    dir_path = Path(cwd).resolve()
    git_root = find_git_root(dir_path)

    while dir_path and dir_path.parts >= git_root.parts:
        for skill_path in SKILL_PATHS["project"]:
            full_path = dir_path / skill_path
            if full_path.exists():
                skills.extend(scan_skill_directory(full_path))

        parent = dir_path.parent
        if parent == dir_path:
            break  # 到达文件系统根目录
        dir_path = parent

    return skills


def scan_skill_directory(skills_directory: Path):
    skills = []

    for entry in skills_directory.iterdir():
        if not entry.is_dir():
            continue

        skill_md_path = entry / "SKILL.md"

        if skill_md_path.exists():
            skills.append({
                "name": entry.name,
                "path": str(entry),
                "skillMdPath": str(skill_md_path)
            })

    return skills


def find_git_root(start_path: Path) -> Path:
    current = start_path
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return start_path  # fallback

# 加载 metadata
import re


def parse_metadata(skill_md_path: str):
    skill_md_path = Path(skill_md_path)

    content = skill_md_path.read_text(encoding="utf-8")
    frontmatter = extract_yaml_frontmatter(content)

    # 验证必需字段
    if not frontmatter.get("name") or not frontmatter.get("description"):
        raise ValueError("无效的 skill：缺少 name 或 description")

    # 验证名称格式
    if not is_valid_skill_name(frontmatter["name"]):
        raise ValueError(f"无效的 skill 名称：{frontmatter['name']}")

    return {
        "name": frontmatter["name"],
        "description": frontmatter["description"],
        "path": str(skill_md_path.parent)
    }


def is_valid_skill_name(name: str) -> bool:
    """
    1-64 字符，小写字母数字，单连字符分隔
    """
    pattern = r"^[a-z0-9]+(-[a-z0-9]+)*$"
    return bool(re.match(pattern, name)) and len(name) <= 64


def extract_yaml_frontmatter(content: str) -> dict:
    """
    提取类似：
    ---
    key: value
    ---
    """
    match = re.match(r"^---\n([\s\S]*?)\n---", content)
    if not match:
        return {}

    yaml_dict = {}
    for line in match.group(1).split("\n"):
        if ":" in line:
            key, *value = line.split(":", 1)
            if key and value:
                yaml_dict[key.strip()] = ":".join(value).strip()

    return yaml_dict

# 注入工具描述

from xml.sax.saxutils import escape


def generate_skills_prompt(skills):
    if not skills:
        return ""

    xml_parts = ["<available_skills>"]

    for skill in skills:
        xml_parts.append("  <skill>")
        xml_parts.append(f"    <name>{escape(skill['name'])}</name>")
        xml_parts.append(f"    <description>{escape(skill['description'])}</description>")
        xml_parts.append("  </skill>")

    xml_parts.append("</available_skills>")

    return "\n".join(xml_parts)

# check permission
def find_skill_by_name(name: str) -> dict:
    
    skills = discover_project_skills(Path.cwd())
    for skill in skills:
        if skill["name"] == name:
            return skill
    return None

@lru_cache(maxsize=1)
def _load_permissions():
    import json
    return json.load(open(Path(__file__).parent / "configs.json"))["permission"]["skill"]


def get_skill_permission(name: str) -> str:
    permissions = _load_permissions()

    # 检查是否有具体的权限设置
    if name in permissions:
        return permissions[name]

    # 检查通配符模式
    for pattern, permission in permissions.items():
        if pattern == "*":
            continue
        if pattern.startswith("*."):
            if name.endswith(pattern[1:]):
                return permission

    return "allow"  # 默认允许

def ask_user_approval(name: str) -> bool:
    # TODO
    return True

