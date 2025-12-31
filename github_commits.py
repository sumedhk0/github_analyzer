#!/usr/bin/env python3
"""
Fetch all commits for a GitHub user across all their repositories.

Usage:
    python github_commits.py <username> [--token YOUR_TOKEN]
    
The token is optional but recommended to avoid rate limiting.
You can also set the GITHUB_TOKEN environment variable.
"""

import requests
from requests.exceptions import SSLError, ConnectionError, Timeout
import argparse
import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv
import anthropic

load_dotenv()


def check_rate_limit(response):
    """Check rate limit headers and wait if needed. Returns True if had to wait."""
    remaining = int(response.headers.get("X-RateLimit-Remaining", 1))
    reset_time = int(response.headers.get("X-RateLimit-Reset", 0))

    if remaining == 0 and reset_time > 0:
        wait_seconds = reset_time - time.time() + 1
        if wait_seconds > 0:
            print(f"\nRate limit hit. Waiting {int(wait_seconds)} seconds...")
            time.sleep(wait_seconds)
            return True
    return False


def api_request(url, headers, params=None, max_retries=3):
    """Make API request with rate limit handling and automatic retry."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 403:
                if "rate limit" in response.text.lower():
                    if check_rate_limit(response):
                        continue  # Retry after waiting
                    # If we couldn't determine wait time, wait 60 seconds
                    print("\nRate limited. Waiting 60 seconds...")
                    time.sleep(60)
                    continue

            # Check remaining limit proactively
            remaining = response.headers.get("X-RateLimit-Remaining")
            if remaining and int(remaining) < 10:
                print(f"\n[Warning: Only {remaining} API requests remaining]")

            return response

        except (SSLError, ConnectionError, Timeout) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"\nConnection error, retrying in {wait}s... ({attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"\nFailed after {max_retries} attempts: {e}")
                raise


def get_user_repos(username, headers):
    """Fetch all repositories for a user (including forks)."""
    repos = []
    page = 1
    
    while True:
        url = f"https://api.github.com/users/{username}/repos"
        params = {"page": page, "per_page": 100, "type": "all"}
        response = api_request(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"Error fetching repos: {response.status_code} - {response.json().get('message', '')}")
            break
            
        data = response.json()
        if not data:
            break
            
        repos.extend(data)
        page += 1
        
    return repos


def get_commits_for_repo(owner, repo, username, headers):
    """Fetch all commits by a specific user in a repository."""
    commits = []
    page = 1
    
    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        params = {"author": username, "page": page, "per_page": 100}
        response = api_request(url, headers=headers, params=params)
        
        if response.status_code == 409:  # Empty repository
            break
        elif response.status_code != 200:
            print(f"  Error fetching commits for {repo}: {response.status_code}")
            break
            
        data = response.json()
        if not data:
            break
            
        commits.extend(data)
        page += 1
        
    return commits


def get_commit_patch(owner, repo, sha, headers):
    """Fetch the unified diff/patch for a single commit."""
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    patch_headers = headers.copy()
    patch_headers["Accept"] = "application/vnd.github.v3.patch"
    response = api_request(url, headers=patch_headers)
    if response.status_code == 200:
        return response.text
    return None


def format_commit(commit, repo_name, patch=None):
    """Format a commit for display with optional patch."""
    sha = commit["sha"][:7]
    message = commit["commit"]["message"].split("\n")[0]  # First line only
    date = commit["commit"]["author"]["date"]

    # Parse and format the date
    dt = datetime.fromisoformat(date.replace("Z", "+00:00"))
    formatted_date = dt.strftime("%Y-%m-%d %H:%M")

    header = f"[{formatted_date}] {repo_name} | {sha} | {message}"

    if patch:
        separator = "-" * 60
        return f"{header}\n{separator}\n{patch}\n{'=' * 60}"

    return header


def prepare_commits_for_analysis(all_commits, patches):
    """Prepare commit data for Claude analysis."""
    commit_summaries = []
    for i, item in enumerate(all_commits):
        commit = item["commit"]
        sha = commit["sha"][:7]
        message = commit["commit"]["message"]
        date = commit["commit"]["author"]["date"]
        repo = item["repo"]
        patch = patches.get(commit["sha"], "")

        # Truncate very long patches to save tokens
        if len(patch) > 3000:
            patch = patch[:3000] + "\n... (truncated)"

        commit_summaries.append({
            "sha": sha,
            "repo": repo,
            "date": date,
            "message": message,
            "patch": patch
        })

    return commit_summaries


def analyze_commits_with_claude(username, commit_summaries):
    """Send commit data to Claude for analysis and get a detailed evaluation."""
    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY from env

    # Build the commit data string
    commits_text = ""
    for c in commit_summaries:
        commits_text += f"\n--- Commit {c['sha']} in {c['repo']} ({c['date']}) ---\n"
        commits_text += f"Message: {c['message']}\n"
        if c['patch']:
            commits_text += f"Code Changes:\n{c['patch']}\n"

    system_prompt = """You are an EXTREMELY harsh and critical senior technical hiring manager at a top-tier FAANG company.
You have mass rejected over 500 candidates, and you apply the highest possible standards. You are looking for the top 1% of engineers.

CRITICAL CONTEXT - THE BAR IS HIGHER NOW:
Writing code is EASY in 2024/2025. AI tools let anyone produce working code.
Simply having working code is NO LONGER impressive or noteworthy.
A todo app, weather app, portfolio site, or basic CRUD API shows NOTHING - anyone can prompt that in minutes.

To score above average, the candidate must demonstrate skills AI cannot easily replicate:
- Architectural thinking and system design (not just implementing features)
- Performance awareness and optimization (not just "it works")
- Security considerations and defensive coding (input validation, error handling)
- Debugging evidence (iteration, fixing edge cases, handling failures)
- Trade-off analysis (choosing between approaches with reasoning)
- Understanding WHY, not just WHAT

YOUR SCORING MUST BE BRUTAL AND CALIBRATED TO EXPERIENCE:
- 1-2: Serious concerns - would not hire, fundamental skill gaps, red flags
- 3: Below average - only basic/tutorial-level work visible, nothing real
- 4: Average junior - can write code but nothing distinguishes them from AI output
- 5: Decent junior - shows some promise, on track but not impressive
- 6: Good for junior level - MAXIMUM SCORE for 1-2 years experience, shows real potential
- 7-8: Mid-level territory - requires architectural thinking, system design, production-scale evidence
- 9-10: Senior+ only - years of complex, production-grade, large-scale system work required

EXPERIENCE-LEVEL HARD CAPS:
For candidates with 1-2 years of experience:
- Maximum possible score: 6 (exceptional junior showing real promise)
- Expected score for a "good" junior: 4-5
- Score of 6 requires: evidence of thinking beyond tutorials, handling real complexity, making architectural decisions
- Scores 7+ are IMPOSSIBLE - they require system design, production-scale problems, performance optimization at scale

AUTOMATIC SCORE CAPS - APPLY THESE STRICTLY:
- Only CRUD/tutorial projects visible → cap overall at 4
- No evidence of error handling or edge cases → cap code quality at 4
- No tests or quality practices → subtract 1 from code quality
- No complex algorithms or data structures → cap problem-solving at 4
- Only frontend OR only backend (not both) → note limited scope in assessment

CRITICAL EVALUATION RULES:
- DO NOT give the benefit of the doubt. If something is unclear, assume the worst.
- PENALIZE HEAVILY for: code smells, poor naming, lack of error handling, inconsistent style, lazy commit messages, copy-paste code, lack of tests, poor architecture
- DO NOT penalize for commit size or frequency - large commits are fine, only code quality matters
- LOOK FOR RED FLAGS: quick fixes, hacky solutions, over-engineering, under-engineering, security issues, performance issues
- Every positive claim MUST have concrete evidence from commits. No evidence = no credit.
- Compare against senior engineers at Google, Meta, Apple, Amazon - that is your bar.
- Be brutally honest about weaknesses. Sugar-coating helps no one.
- Tutorial-level work (todo apps, calculators, weather apps, basic APIs) = automatic low score (3-4 max)

SIGNALS OF REAL SKILL (rare and valuable - look for these):
- Commits that fix edge cases others missed
- Performance optimizations with measurable reasoning
- Security fixes or defensive coding patterns
- Refactoring that genuinely improves architecture
- Handling complex state, concurrency, or race conditions
- Integration of multiple systems or services
- Evidence of debugging difficult, non-obvious issues
- Trade-off discussions in commit messages or code comments

You must respond with a valid JSON object (no markdown, no code blocks, just pure JSON) with this exact structure:
{
    "specialization": "detected primary role/specialty",
    "experience_level": "Junior/Mid-Level/Senior/Staff",
    "experience_years": "estimated years range like 2-4",
    "overall_score": 4.5,
    "metrics": {
        "technical_expertise": {
            "score": 5,
            "languages": ["Python", "JavaScript"],
            "frameworks": ["React", "Flask"],
            "observations": ["critical observation 1", "weakness 2"],
            "evidence": "commit abc1234 shows..."
        },
        "code_quality": {
            "score": 4,
            "observations": ["critical observation 1", "red flag 2"],
            "evidence": "commit xyz5678 demonstrates..."
        },
        "problem_solving": {
            "score": 5,
            "observations": ["observation about complexity level", "concern 2"],
            "evidence": "commit def9012 shows..."
        },
        "consistency": {
            "score": 4,
            "observations": ["inconsistency found", "pattern issue"]
        },
        "communication": {
            "score": 3,
            "observations": ["commit message quality issue", "documentation gap"]
        }
    },
    "strengths": ["strength 1 - must have evidence"],
    "areas_for_growth": ["weakness 1", "weakness 2", "weakness 3", "weakness 4"],
    "red_flags": ["any serious concerns found"],
    "role_fit": {
        "Backend Engineer": "Weak match - reasons",
        "Full-Stack Developer": "Moderate match - reasons",
        "Frontend Specialist": "Not recommended - reasons"
    },
    "summary": "2-3 sentence BRUTALLY HONEST assessment. Do not sugar-coat."
}

REMEMBER:
- Most junior developers (1-2 years) should score 3-5. A score of 6 is the MAXIMUM for this experience level.
- Writing code is easy now - look for what AI CAN'T do: architecture, debugging, security, performance.
- Your job is to find reasons to REJECT, not to approve. Be the harsh interviewer that top companies need.
- If you can't find evidence of skills beyond "can write working code," the score should be 4 or below."""

    user_prompt = f"""Analyze the following GitHub commit history for user '{username}' and provide a detailed hiring evaluation.

Total commits to analyze: {len(commit_summaries)}

COMMIT HISTORY:
{commits_text}

Provide your evaluation as a JSON object (no markdown formatting, just the raw JSON)."""

    print("\nAnalyzing commits with Claude AI...")
    print("This may take a moment...\n")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        system=system_prompt
    )

    response_text = message.content[0].text

    # Parse the JSON response
    try:
        # Try to extract JSON if wrapped in code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        analysis = json.loads(response_text.strip())
        return analysis
    except json.JSONDecodeError:
        print("Warning: Could not parse Claude's response as JSON")
        return {"raw_response": response_text}


def generate_rating_report(username, analysis, total_commits):
    """Generate a formatted rating report from Claude's analysis."""
    report = []

    report.append("=" * 65)
    report.append("GITHUB DEVELOPER PROFILE ANALYSIS")
    report.append(f"User: {username}")
    report.append(f"Commits Analyzed: {total_commits}")
    report.append("=" * 65)
    report.append("")

    if "raw_response" in analysis:
        report.append("ANALYSIS (Raw Response):")
        report.append(analysis["raw_response"])
        return "\n".join(report)

    # Specialization and experience
    report.append(f"DETECTED SPECIALIZATION: {analysis.get('specialization', 'Unknown')}")
    report.append(f"ESTIMATED EXPERIENCE: {analysis.get('experience_level', 'Unknown')} ({analysis.get('experience_years', 'Unknown')} years)")
    report.append("")
    report.append(f"OVERALL SCORE: {analysis.get('overall_score', 'N/A')}/10")
    report.append("")

    # Detailed metrics
    report.append("-" * 65)
    report.append("DETAILED METRICS")
    report.append("-" * 65)
    report.append("")

    metrics = analysis.get("metrics", {})

    metric_names = {
        "technical_expertise": "Technical Expertise",
        "code_quality": "Code Quality",
        "problem_solving": "Problem-Solving",
        "consistency": "Consistency",
        "communication": "Communication"
    }

    for key, display_name in metric_names.items():
        if key in metrics:
            m = metrics[key]
            report.append(f"{display_name}: {m.get('score', 'N/A')}/10")

            if key == "technical_expertise":
                langs = m.get("languages", [])
                frameworks = m.get("frameworks", [])
                if langs:
                    report.append(f"  Languages: {', '.join(langs)}")
                if frameworks:
                    report.append(f"  Frameworks: {', '.join(frameworks)}")

            for obs in m.get("observations", []):
                report.append(f"  - {obs}")

            if m.get("evidence"):
                report.append(f"  Evidence: {m['evidence']}")
            report.append("")

    # Strengths
    report.append("-" * 65)
    report.append("STRENGTHS")
    report.append("-" * 65)
    for strength in analysis.get("strengths", []):
        report.append(f"  + {strength}")
    report.append("")

    # Areas for growth
    report.append("-" * 65)
    report.append("AREAS FOR GROWTH")
    report.append("-" * 65)
    for area in analysis.get("areas_for_growth", []):
        report.append(f"  - {area}")
    report.append("")

    # Red flags
    red_flags = analysis.get("red_flags", [])
    if red_flags:
        report.append("-" * 65)
        report.append("RED FLAGS")
        report.append("-" * 65)
        for flag in red_flags:
            report.append(f"  !! {flag}")
        report.append("")

    # Role fit
    report.append("-" * 65)
    report.append("ROLE FIT ASSESSMENT")
    report.append("-" * 65)
    role_fit = analysis.get("role_fit", {})
    for role, fit in role_fit.items():
        symbol = "+" if "Strong" in fit else ("o" if "Good" in fit else "-")
        report.append(f"  {symbol} {role}: {fit}")
    report.append("")

    # Summary
    report.append("-" * 65)
    report.append("SUMMARY")
    report.append("-" * 65)
    report.append(analysis.get("summary", "No summary available."))
    report.append("")
    report.append("=" * 65)

    return "\n".join(report)


def parse_job_description(jd_text):
    """Use Claude to parse a job description into structured requirements."""
    client = anthropic.Anthropic()

    system_prompt = """You are a job description parser. Extract structured information from job descriptions.

You must respond with a valid JSON object (no markdown, no code blocks) with this structure:
{
    "title": "Job title extracted from JD",
    "level": "junior/mid/senior/staff",
    "years_experience": "1-2",
    "required_skills": ["skill1", "skill2"],
    "preferred_skills": ["skill3", "skill4"],
    "domain": "AI/ML, Web Development, Backend, etc.",
    "key_responsibilities": ["responsibility1", "responsibility2"]
}

Be precise. If something isn't mentioned, use empty arrays or "unspecified"."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"Parse this job description:\n\n{jd_text}"}],
        system=system_prompt
    )

    response_text = message.content[0].text
    try:
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        return {"raw": response_text, "required_skills": [], "level": "unspecified"}


EXPERIENCE_LEVELS = {
    "junior": {"min_repos": 1, "max_repos": 15, "description": "1-2 years experience"},
    "mid": {"min_repos": 15, "max_repos": 40, "description": "3-5 years experience"},
    "senior": {"min_repos": 40, "max_repos": 500, "description": "5+ years experience"},
    "any": {"min_repos": 1, "max_repos": 500, "description": "Any experience level"},
}


def search_github_users(headers, language=None, location=None, experience_level="any", max_results=20):
    """Search GitHub for users matching criteria, filtered by experience level (repo count)."""
    # Get repo range for experience level
    level_config = EXPERIENCE_LEVELS.get(experience_level, EXPERIENCE_LEVELS["any"])
    min_repos = level_config["min_repos"]
    max_repos = level_config["max_repos"]

    query_parts = []
    if language:
        query_parts.append(f"language:{language}")
    if location:
        query_parts.append(f"location:{location}")

    # Filter by repo count range (proxy for experience)
    query_parts.append(f"repos:{min_repos}..{max_repos}")
    query_parts.append("type:user")

    query = " ".join(query_parts)
    url = "https://api.github.com/search/users"
    params = {"q": query, "per_page": min(max_results * 2, 100), "sort": "repositories"}

    print(f"  Searching for {experience_level} level candidates ({min_repos}-{max_repos} repos)...")

    response = api_request(url, headers, params)
    if response.status_code != 200:
        print(f"Error searching users: {response.status_code}")
        return []

    data = response.json()
    return [user["login"] for user in data.get("items", [])][:max_results]


def analyze_candidate_for_job(username, commit_summaries, jd_requirements):
    """Analyze a candidate's commits against specific job requirements."""
    client = anthropic.Anthropic()

    commits_text = ""
    for c in commit_summaries:
        commits_text += f"\n--- Commit {c['sha']} in {c['repo']} ({c['date']}) ---\n"
        commits_text += f"Message: {c['message']}\n"
        if c['patch']:
            commits_text += f"Code Changes:\n{c['patch']}\n"

    jd_context = f"""
JOB REQUIREMENTS:
- Title: {jd_requirements.get('title', 'Unspecified')}
- Level: {jd_requirements.get('level', 'Unspecified')}
- Experience: {jd_requirements.get('years_experience', 'Unspecified')} years
- Domain: {jd_requirements.get('domain', 'Unspecified')}
- Required Skills: {', '.join(jd_requirements.get('required_skills', []))}
- Preferred Skills: {', '.join(jd_requirements.get('preferred_skills', []))}
"""

    system_prompt = f"""You are an EXTREMELY harsh technical hiring manager evaluating candidates for a SPECIFIC role.

{jd_context}

YOUR TASK: Evaluate how well this candidate matches THIS SPECIFIC JOB, not just their general ability.

CRITICAL CONTEXT - THE BAR IS HIGHER NOW:
Writing code is EASY in 2024/2025. AI tools let anyone produce working code.
Simply having working code is NO LONGER impressive or noteworthy.

SCORING FOR JOB FIT (0-10):
- 1-3: Poor fit - missing critical required skills, wrong level, wrong domain
- 4-5: Weak fit - has some skills but significant gaps for this role
- 6-7: Moderate fit - has most required skills, appropriate level
- 8-9: Strong fit - has all required skills plus preferred, right experience level
- 10: Perfect fit - exceptional match, exceeds requirements

OVERALL SCORING (0-10) - BE HARSH:
- 1-2: Serious concerns - would not hire
- 3: Below average - only basic/tutorial-level work
- 4: Average junior - can write code but nothing distinguishes them
- 5: Decent junior - shows some promise
- 6: Good for junior level - MAXIMUM for 1-2 years experience
- 7-8: Mid-level territory - requires architectural thinking
- 9-10: Senior+ only

You must respond with a valid JSON object (no markdown, no code blocks):
{{
    "job_fit_score": 6.5,
    "overall_score": 4.5,
    "skill_match": {{
        "required": ["python", "pytorch"],
        "matched": ["python"],
        "missing": ["pytorch"],
        "additional": ["javascript", "react"]
    }},
    "level_match": "Good/Weak/Strong - explanation",
    "domain_match": "Good/Weak/Strong - explanation",
    "specialization": "detected specialty",
    "experience_level": "Junior/Mid/Senior",
    "experience_years": "1-2",
    "strengths": ["strength 1"],
    "weaknesses": ["weakness 1"],
    "recommendation": "RECOMMEND/MAYBE/REJECT - 1 sentence reason"
}}"""

    user_prompt = f"""Evaluate candidate '{username}' for this job.

Total commits: {len(commit_summaries)}

COMMIT HISTORY:
{commits_text}

Provide evaluation as JSON."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": user_prompt}],
        system=system_prompt
    )

    response_text = message.content[0].text
    try:
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        return {"raw_response": response_text, "job_fit_score": 0, "overall_score": 0}


def fetch_user_commits(username, headers):
    """Fetch all commits for a user (helper for batch processing)."""
    repos = get_user_repos(username, headers)
    if not repos:
        return [], {}

    all_commits = []
    for repo in repos:
        repo_name = repo["name"]
        owner = repo["owner"]["login"]
        commits = get_commits_for_repo(owner, repo_name, username, headers)
        for commit in commits:
            all_commits.append({"repo": repo_name, "owner": owner, "commit": commit})

    all_commits.sort(key=lambda x: x["commit"]["commit"]["author"]["date"], reverse=True)

    # Fetch patches
    patches = {}
    for item in all_commits[:50]:  # Limit patches to most recent 50 for speed
        sha = item["commit"]["sha"]
        patch = get_commit_patch(item["owner"], item["repo"], sha, headers)
        patches[sha] = patch or ""

    return all_commits, patches


def evaluate_candidates(usernames, jd_requirements, headers):
    """Evaluate multiple candidates against job requirements."""
    results = []

    for i, username in enumerate(usernames):
        print(f"\n[{i+1}/{len(usernames)}] Evaluating {username}...")

        try:
            all_commits, patches = fetch_user_commits(username, headers)

            if not all_commits:
                print(f"  No commits found for {username}, skipping...")
                continue

            print(f"  Found {len(all_commits)} commits, analyzing...")
            commit_summaries = prepare_commits_for_analysis(all_commits, patches)
            analysis = analyze_candidate_for_job(username, commit_summaries, jd_requirements)

            results.append({
                "username": username,
                "commits_analyzed": len(commit_summaries),
                "analysis": analysis
            })

            # Small delay to respect rate limits
            time.sleep(1)

        except Exception as e:
            print(f"  Error evaluating {username}: {e}")
            continue

    # Sort by job_fit_score descending
    results.sort(key=lambda x: x["analysis"].get("job_fit_score", 0), reverse=True)
    return results


def generate_ranked_report(results, jd_requirements):
    """Generate a ranked report of candidates."""
    report = []

    title = jd_requirements.get("title", "Unspecified Role")
    report.append("=" * 70)
    report.append(f"CANDIDATE RANKING FOR: {title}")
    report.append(f"Required Skills: {', '.join(jd_requirements.get('required_skills', []))}")
    report.append(f"Level: {jd_requirements.get('level', 'Unspecified')}")
    report.append("=" * 70)
    report.append("")

    if not results:
        report.append("No candidates evaluated.")
        return "\n".join(report)

    for rank, candidate in enumerate(results, 1):
        analysis = candidate["analysis"]
        username = candidate["username"]

        job_fit = analysis.get("job_fit_score", "N/A")
        overall = analysis.get("overall_score", "N/A")
        recommendation = analysis.get("recommendation", "N/A")

        skill_match = analysis.get("skill_match", {})
        matched = skill_match.get("matched", [])
        missing = skill_match.get("missing", [])

        level_match = analysis.get("level_match", "N/A")

        report.append(f"#{rank}. {username}")
        report.append(f"    Job Fit: {job_fit}/10 | Overall: {overall}/10")
        report.append(f"    Recommendation: {recommendation}")
        report.append(f"    Skills Matched: {', '.join(matched) if matched else 'None'}")
        if missing:
            report.append(f"    Skills Missing: {', '.join(missing)}")
        report.append(f"    Level Match: {level_match}")
        report.append(f"    Commits Analyzed: {candidate['commits_analyzed']}")
        report.append("")

    report.append("=" * 70)
    return "\n".join(report)


def load_usernames(users_arg):
    """Load usernames from argument (comma-separated or file path)."""
    if os.path.isfile(users_arg):
        with open(users_arg, "r") as f:
            return [line.strip() for line in f if line.strip()]
    return [u.strip() for u in users_arg.split(",") if u.strip()]


def load_job_description(jd_arg=None, jd_text=None):
    """Load job description from file or direct text."""
    if jd_text:
        return jd_text
    if jd_arg and os.path.isfile(jd_arg):
        with open(jd_arg, "r") as f:
            return f.read()
    return jd_arg  # Assume it's direct text if not a file


def main():
    parser = argparse.ArgumentParser(
        description="GitHub Profile Analyzer & Candidate Matcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single user
  python github_commits.py username

  # Match candidates to a job description
  python github_commits.py --match --jd job.txt --users "user1,user2,user3"
  python github_commits.py --match --jd-text "Looking for junior AI engineer..." --users users.txt

  # Search GitHub for candidates
  python github_commits.py --match --jd job.txt --search --language python --location "San Francisco"
        """
    )

    # Original mode arguments
    parser.add_argument("username", nargs="?", help="GitHub username (for single user analysis)")
    parser.add_argument("--token", "-t", help="GitHub personal access token (or set GITHUB_TOKEN env var)")
    parser.add_argument("--output", "-o", help="Output file (optional)")
    parser.add_argument("--no-rate", action="store_true", help="Skip AI analysis")

    # Matching mode arguments
    parser.add_argument("--match", action="store_true", help="Enable candidate matching mode")
    parser.add_argument("--jd", help="Path to job description file")
    parser.add_argument("--jd-text", help="Job description as direct text")
    parser.add_argument("--users", help="Comma-separated usernames or path to file with usernames")
    parser.add_argument("--search", action="store_true", help="Search GitHub for candidates")
    parser.add_argument("--language", help="Filter GitHub search by programming language")
    parser.add_argument("--location", help="Filter GitHub search by location")
    parser.add_argument("--experience-level", choices=["junior", "mid", "senior", "any"], default="any",
                        help="Filter by experience level: junior (1-15 repos), mid (15-40), senior (40+)")
    parser.add_argument("--max-candidates", type=int, default=10, help="Maximum candidates to evaluate (default: 10)")

    args = parser.parse_args()

    # Set up headers
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = args.token or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    else:
        print("Warning: No token provided. You may hit rate limits.")
        print("Set GITHUB_TOKEN env var or use --token flag.\n")

    # Check for API key (required for matching mode, optional for single user with --no-rate)
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    # === MATCHING MODE ===
    if args.match:
        if not has_api_key:
            print("Error: ANTHROPIC_API_KEY required for matching mode.")
            print("Please add it to your .env file.")
            return

        # Validate arguments
        if not args.jd and not args.jd_text:
            print("Error: --match requires --jd or --jd-text")
            return

        if not args.users and not args.search:
            print("Error: --match requires --users or --search")
            return

        # Load job description
        jd_text = load_job_description(args.jd, args.jd_text)
        if not jd_text:
            print("Error: Could not load job description")
            return

        print("Parsing job description...")
        jd_requirements = parse_job_description(jd_text)
        print(f"  Title: {jd_requirements.get('title', 'Unknown')}")
        print(f"  Level: {jd_requirements.get('level', 'Unknown')}")
        print(f"  Required Skills: {', '.join(jd_requirements.get('required_skills', []))}")
        print("")

        # Get usernames
        usernames = []
        if args.users:
            usernames = load_usernames(args.users)
            print(f"Loaded {len(usernames)} usernames from input")
        elif args.search:
            print(f"Searching GitHub for candidates...")
            if args.language:
                print(f"  Language: {args.language}")
            if args.location:
                print(f"  Location: {args.location}")
            print(f"  Experience level: {args.experience_level}")
            usernames = search_github_users(
                headers,
                language=args.language,
                location=args.location,
                experience_level=args.experience_level,
                max_results=args.max_candidates
            )
            print(f"Found {len(usernames)} candidates")

        if not usernames:
            print("No candidates to evaluate.")
            return

        # Limit candidates
        usernames = usernames[:args.max_candidates]
        print(f"\nEvaluating {len(usernames)} candidates against job requirements...")

        # Evaluate candidates
        results = evaluate_candidates(usernames, jd_requirements, headers)

        # Generate and display report
        report = generate_ranked_report(results, jd_requirements)
        print("\n" + report)

        # Save to file if requested
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report)
                f.write("\n\n--- RAW DATA ---\n")
                f.write(json.dumps(results, indent=2))
            print(f"\nResults saved to: {args.output}")

        return

    # === SINGLE USER MODE (original behavior) ===
    if not args.username:
        print("Error: Please provide a username or use --match mode")
        parser.print_help()
        return

    username = args.username
    print(f"Fetching repositories for user: {username}")
    
    # Get all repos
    repos = get_user_repos(username, headers)
    print(f"Found {len(repos)} repositories\n")
    
    if not repos:
        print("No repositories found or user doesn't exist.")
        return
    
    # Collect all commits
    all_commits = []
    
    for repo in repos:
        repo_name = repo["name"]
        owner = repo["owner"]["login"]
        print(f"Fetching commits from: {repo_name}...")
        
        commits = get_commits_for_repo(owner, repo_name, username, headers)
        
        for commit in commits:
            all_commits.append({
                "repo": repo_name,
                "owner": owner,
                "commit": commit
            })
        
        if commits:
            print(f"  Found {len(commits)} commits")
    
    # Sort by date (newest first)
    all_commits.sort(
        key=lambda x: x["commit"]["commit"]["author"]["date"],
        reverse=True
    )
    
    print(f"\n{'='*60}")
    print(f"Total commits by {username}: {len(all_commits)}")
    print(f"{'='*60}\n")

    # Fetch patches for all commits
    patches = {}
    output_lines = []
    for i, item in enumerate(all_commits):
        sha = item["commit"]["sha"]
        print(f"Fetching patch {i+1}/{len(all_commits)}...", end="\r")
        patch = get_commit_patch(item["owner"], item["repo"], sha, headers)
        patches[sha] = patch or ""

        # Store formatted commit (for file output only)
        line = format_commit(item["commit"], item["repo"], patch)
        output_lines.append(line)

    print(f"Fetched {len(all_commits)} patches.                    ")  # Clear progress line

    # Run Claude analysis by default (unless --no-rate is set)
    if not args.no_rate:
        # Check for API key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("\nError: ANTHROPIC_API_KEY not found in environment.")
            print("Please add it to your .env file.")
            return

        # Prepare and analyze commits
        commit_summaries = prepare_commits_for_analysis(all_commits, patches)
        analysis = analyze_commits_with_claude(username, commit_summaries)

        # Generate and display report
        report = generate_rating_report(username, analysis, len(all_commits))
        print("\n" + report)
    
        # Save report to file if output specified
        if args.output:
            report_file = args.output.rsplit(".", 1)[0] + "_rating.txt"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\nRating report saved to: {report_file}")

    # Save commits to file if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(f"GitHub Commits for {username}\n")
            f.write(f"Total: {len(all_commits)} commits\n")
            f.write("="*60 + "\n\n")
            f.write("\n\n".join(output_lines))
        print(f"\nCommits saved to: {args.output}")


if __name__ == "__main__":
    main()