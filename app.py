#!/usr/bin/env python3
"""
Flask web frontend for GitHub Profile Analyzer & Candidate Matcher.

Usage:
    python app.py

Then open http://localhost:5000 in your browser.
"""

from flask import Flask, render_template, request, jsonify, Response
import os
import json
from dotenv import load_dotenv

# Import functions from the existing CLI tool
from github_commits import (
    get_user_repos,
    get_commits_for_repo,
    get_commit_patch,
    prepare_commits_for_analysis,
    analyze_commits_with_claude,
    generate_rating_report,
    parse_job_description,
    search_github_users,
    analyze_candidate_for_job,
    fetch_user_commits,
    evaluate_candidates,
    generate_ranked_report,
)

load_dotenv()

app = Flask(__name__)

# Set up GitHub headers
def get_headers():
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


@app.route("/")
def index():
    """Home page - single user analysis form."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze a single GitHub user."""
    username = request.form.get("username", "").strip()
    if not username:
        return render_template("profile.html", error="Please enter a username")

    headers = get_headers()

    try:
        # Fetch repos and commits
        repos = get_user_repos(username, headers)
        if not repos:
            return render_template("profile.html", error=f"No repositories found for {username}")

        all_commits = []
        for repo in repos:
            repo_name = repo["name"]
            owner = repo["owner"]["login"]
            commits = get_commits_for_repo(owner, repo_name, username, headers)
            for commit in commits:
                all_commits.append({"repo": repo_name, "owner": owner, "commit": commit})

        if not all_commits:
            return render_template("profile.html", error=f"No commits found for {username}")

        # Sort by date
        all_commits.sort(key=lambda x: x["commit"]["commit"]["author"]["date"], reverse=True)

        # Fetch patches (limit to 50 for speed)
        patches = {}
        for item in all_commits[:50]:
            sha = item["commit"]["sha"]
            patch = get_commit_patch(item["owner"], item["repo"], sha, headers)
            patches[sha] = patch or ""

        # Analyze with Claude
        commit_summaries = prepare_commits_for_analysis(all_commits, patches)
        analysis = analyze_commits_with_claude(username, commit_summaries)

        return render_template(
            "profile.html",
            username=username,
            analysis=analysis,
            total_commits=len(all_commits),
            total_repos=len(repos)
        )

    except Exception as e:
        return render_template("profile.html", error=str(e))


@app.route("/match")
def match_form():
    """Job matching form page."""
    return render_template("match.html")


@app.route("/match", methods=["POST"])
def match():
    """Match candidates against a job description."""
    jd_text = request.form.get("job_description", "").strip()
    users_text = request.form.get("usernames", "").strip()

    if not jd_text:
        return render_template("match.html", error="Please enter a job description")
    if not users_text:
        return render_template("match.html", error="Please enter candidate usernames")

    headers = get_headers()

    try:
        # Parse job description
        jd_requirements = parse_job_description(jd_text)

        # Parse usernames
        usernames = [u.strip() for u in users_text.replace("\n", ",").split(",") if u.strip()]

        if not usernames:
            return render_template("match.html", error="No valid usernames provided")

        # Evaluate candidates
        results = evaluate_candidates(usernames, jd_requirements, headers)

        return render_template(
            "results.html",
            jd_requirements=jd_requirements,
            results=results,
            mode="match"
        )

    except Exception as e:
        return render_template("match.html", error=str(e))


@app.route("/search")
def search_form():
    """GitHub search form page."""
    return render_template("search.html")


@app.route("/search", methods=["POST"])
def search():
    """Search GitHub for candidates and match against JD."""
    jd_text = request.form.get("job_description", "").strip()
    language = request.form.get("language", "").strip() or None
    location = request.form.get("location", "").strip() or None
    experience_level = request.form.get("experience_level", "any").strip()
    max_candidates = int(request.form.get("max_candidates", 10))

    if not jd_text:
        return render_template("search.html", error="Please enter a job description")

    headers = get_headers()

    try:
        # Parse job description
        jd_requirements = parse_job_description(jd_text)

        # Search GitHub for users filtered by experience level
        usernames = search_github_users(
            headers,
            language=language,
            location=location,
            experience_level=experience_level,
            max_results=max_candidates
        )

        if not usernames:
            return render_template("search.html", error="No candidates found matching search criteria")

        # Evaluate candidates
        results = evaluate_candidates(usernames[:max_candidates], jd_requirements, headers)

        return render_template(
            "results.html",
            jd_requirements=jd_requirements,
            results=results,
            mode="search",
            search_params={"language": language, "location": location, "experience_level": experience_level}
        )

    except Exception as e:
        return render_template("search.html", error=str(e))


@app.route("/api/export", methods=["POST"])
def export_results():
    """Export results as JSON."""
    data = request.get_json()
    return Response(
        json.dumps(data, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment;filename=results.json"}
    )


if __name__ == "__main__":
    # Check for required environment variables
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set. Analysis features will not work.")
    if not os.environ.get("GITHUB_TOKEN"):
        print("Warning: GITHUB_TOKEN not set. You may hit rate limits.")

    print("\nStarting GitHub Analyzer Web UI...")
    print("Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
