from pathlib import Path
import subprocess

from utils import find_skill_by_name, get_skill_permission, ask_user_approval


async def load_skill(name: str) -> str | dict:
    """
    Load a skill by name and return its SKILL.md content.

    Args:
        name: The skill identifier to load.

    Returns:
        A dict with 'error' key if failed, otherwise the raw SKILL.md content as string.

    Permission Flow:
        deny -> immediately reject
        ask -> prompt user approval before loading
        allow -> load directly
    """
    skill = find_skill_by_name(name)
    if not skill:
        return {"error": f"Skill not found: {name}"}

    # Check permission level for this skill
    permission = get_skill_permission(name)

    if permission == "deny":
        return {"error": f"Access denied: {name}"}

    if permission == "ask":
        approved = await ask_user_approval(name)
        if not approved:
            return {"error": f"User rejected access to skill: {name}"}

    content = Path(skill["skillMdPath"]).read_text(encoding="utf-8")
    return content


def execute_skill_script(skill_name: str, script_path: str, args: list):
    """
    Execute a script under the given skill's scripts/ directory.

    Args:
        skill_name: Name of the skill (used for path resolution).
        script_path: Relative path within the scripts/ directory, e.g. "run.sh".
        args: Command-line arguments passed to the script.

    Returns:
        A dict with returncode, stdout, and stderr.

    Security:
        Resolves the full path and verifies it stays within the skill_root
        to prevent directory traversal attacks.
    """
    skill = find_skill_by_name(skill_name)
    if not skill:
        raise ValueError(f"Skill not found: {skill_name}")

    skill_root = Path(skill["path"]).resolve()
    full_path = (skill_root / "scripts" / script_path).resolve()

    # Security check: ensure resolved path is still under skill_root (prevents symlink escape)
    if not str(full_path).startswith(str(skill_root)):
        raise ValueError("Invalid script path: escape attempt detected")

    # Execute synchronously, similar to Node's spawnSync
    result = subprocess.run(
        [str(full_path)] + args,
        capture_output=True,
        text=True,
        timeout=60  # prevent indefinite blocking
    )

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }


def load_reference(skill_name: str, ref_path: str) -> str:
    """
    Load a reference file from the skill's references/ directory.

    Args:
        skill_name: Name of the skill (used for path resolution).
        ref_path: Relative path within the references/ directory.

    Returns:
        The raw text content of the reference file.

    Raises:
        ValueError: If the skill does not exist.
    """
    skill = find_skill_by_name(skill_name)
    if not skill:
        raise ValueError(f"Skill not found: {skill_name}")

    skill_root = Path(skill["path"]).resolve()
    full_path = (skill_root / "references" / ref_path).resolve()

    # Security check: prevent directory traversal via ".."
    if not str(full_path).startswith(str(skill_root)):
        raise ValueError("Invalid reference path: escape attempt detected")

    return full_path.read_text(encoding="utf-8")