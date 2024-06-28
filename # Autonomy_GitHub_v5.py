
#%%
#%%
import requests  # For making HTTP requests to the GitHub API
import random  # For random sampling of repositories
import os  # For file and directory operations
from git import Repo  # For cloning Git repositories
import pandas as pd  # For creating and manipulating dataframes
import json  # For reading and writing JSON files
import time  # For handling time-related operations
import re


# Refer to the requirements file for dependencies
#%%

GITHUB_API_URL = "https://api.github.com"
TOPIC = "ai-agents"
SAMPLE_SIZE = 10
REPOS_FILE = "repos_list.json"
DOWNLOAD_DIR = "repos"
LLM_TOOLS = {
    "openai": "import openai",
    "ai21_labs": "import ai21",
    "aleph_alpha": "import aleph_alpha",
    "hugging_face": "from transformers import",
    "google_cloud": "from google.cloud import aiplatform",
    "ibm_watson": "from ibm_watson import"
} # can be expanded flexibly


AGENT_LIBRARIES = {
    "dspy": "dspy",
    "langchain": "langchain",
    "llama_index": "llama_index"
}


def main_fetch(topic=TOPIC, sample_size=10, download_dir=DOWNLOAD_DIR):
    url = f"{GITHUB_API_URL}/search/repositories?q=topic:{topic}&per_page=100"
    response = requests.get(url)
    response.raise_for_status()
    repos = response.json()["items"]
    selected_repos = random.sample(repos, sample_size)
    
    with open(download_dir, "w") as f:
        json.dump(selected_repos, f)
    
    print(f"Fetched {len(selected_repos)} repositories and saved to {download_dir}.")

def download_repository(repo_url, download_dir, retries=2):
    repo_name = repo_url.split("/")[-1]
    repo_path = os.path.join(download_dir, repo_name)
    
    if not os.path.exists(repo_path):
        for attempt in range(retries):
            try:
                Repo.clone_from(repo_url, repo_path, env={"GIT_HTTP_MAX_BUFFER": "104857600"})
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {repo_name}: {e}")
                if attempt == retries - 1:
                    print(f"Skipping {repo_name} after {retries} failed attempts.")
                    return None
                time.sleep(5)  # Wait before retrying
    return repo_path

def main_download():
    start_time = time.time()
    
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    
    with open(REPOS_FILE, "r") as f:
        repos = json.load(f)
    
    for repo in repos:
        repo_start_time = time.time()
        print(f"Downloading repository: {repo['name']}...")
        repo_path = download_repository(repo["clone_url"], DOWNLOAD_DIR)
        if repo_path:
            print(f"Downloaded {repo['name']} in {time.time() - repo_start_time:.2f} seconds.")
        else:
            print(f"Failed to download {repo['name']}.")
    
    print(f"Total download time: {time.time() - start_time:.2f} seconds.")


def initialize_dataset():
    columns = ["repository_name", "repository_content", "overall_llm_usage"] + \
              [f"uses_{llm.lower().replace(' ', '_')}" for llm in LLM_TOOLS] + \
              ["calls_specific_repos", "nested_api_calls", "autonomous_data_generation", "goal_directedness", "time_without_human_oversight"]
    return pd.DataFrame(columns=columns)


def read_repository_content(repo_path):
    repo_content = ""
    for root, _, files in os.walk(repo_path):
        for file in files:
            with open(os.path.join(root, file), "r", errors="ignore") as f:
                repo_content += f.read()
    return repo_content



def check_llm_usage(repo_content):
    overall_llm_usage = False
    llm_usage = {}
    llm_snippets = {}

    for llm in LLM_TOOLS:
        llm_key = f"uses_{llm.lower().replace(' ', '_')}"
        if llm in repo_content:  # Change: should be llm_tools instead of llm
            overall_llm_usage = True
            llm_usage[llm_key] = True
            # Extract snippet
            start_idx = repo_content.find(llm)
            snippet_start = max(0, start_idx - 50)
            snippet_end = min(len(repo_content), start_idx + len(llm) + 50)
            snippet = repo_content[snippet_start:snippet_end]
            llm_snippets[f"{llm_key}_snippet"] = snippet
        else:
            llm_usage[llm_key] = False
            llm_snippets[f"{llm_key}_snippet"] = ""

    return overall_llm_usage, llm_usage, llm_snippets

def check_agent_library_usage(repo_content):
    agent_usage = {}
    agent_snippets = {}

    for agent, import_statement in AGENT_LIBRARIES.items():
        agent_key = f"uses_{agent}"
        pattern = re.compile(rf'\b{re.escape(import_statement)}\b', re.IGNORECASE)
        match = pattern.search(repo_content)
        
        if match:
            agent_usage[agent_key] = True
            start_idx = match.start()
            snippet_start = max(0, start_idx - 50)
            snippet_end = min(len(repo_content), start_idx + len(import_statement) + 50)
            snippet = repo_content[snippet_start:snippet_end]
            agent_snippets[f"{agent_key}_snippet"] = snippet
        else:
            agent_usage[agent_key] = False
            agent_snippets[f"{agent_key}_snippet"] = ""

    return agent_usage, agent_snippets

def process_repository(repo):
    repo_name = repo["name"]
    repo_path = os.path.join(DOWNLOAD_DIR, repo_name)
    repo_content = read_repository_content(repo_path)
    
    overall_llm_usage, llm_usage, llm_snippets = check_llm_usage(repo_content)
    print(f"Repository: {repo_name}")
    print("Overall LLM usage:", overall_llm_usage)
    print("Specific LLM usage:", llm_usage)
    print("LLM usage snippets:", llm_snippets)
    print("\n")
    
    agent_usage, agent_snippets = check_agent_library_usage(repo_content)
    overall_agent_usage = any(agent_usage.values())
    print(f"Repository: {repo_name}")
    print("Overall agent library usage:", overall_agent_usage)
    print("Specific Agent library usage:", agent_usage)
    print("Agent library usage snippets:", agent_snippets)
    print("\n")
    
    repo_data = {
        "repository_name": repo_name,
        "repository_content": repo_content,
        "overall_llm_usage": overall_llm_usage,
        **llm_usage,
        **llm_snippets,
        **agent_usage,
        **agent_snippets,
        "overall_agent_usage": overall_agent_usage
    }
    
    return repo_data

def main(sample_size=0, analyse_size=10, csv=False):
    if sample_size > 0:
        main_fetch(sample_size=sample_size)  # Fetch the repositories first
        main_download(sample_size)  # Download the fetched repositories

    dataset = initialize_dataset()
    data_list = []
    llm_usage_count = 0
    agent_usage_count = 0

    for repo_name in os.listdir(DOWNLOAD_DIR)[:analyse_size]:  # Iterate over the directory names up to analyse_size
        repo = {"name": repo_name}  # Create a dictionary with the repo name
        repo_data = process_repository(repo)
        data_list.append(repo_data)
        
        if repo_data["overall_llm_usage"]:
            llm_usage_count += 1
        
        if repo_data["overall_agent_usage"]:
            agent_usage_count += 1

    llm_usage_share = llm_usage_count / analyse_size if analyse_size > 0 else 0
    agent_usage_share = agent_usage_count / analyse_size if analyse_size > 0 else 0

    print(f"Share of repos with LLM usage = Yes: {llm_usage_share}")
    print(f"Share of repos with overall agent library usage = Yes: {agent_usage_share}")

    if csv == True:
        dataset = pd.DataFrame(data_list)
        dataset.to_csv("repository_analysis.csv", index=False)

if __name__ == "__main__":
    main(analyse_size=10)
# %%
