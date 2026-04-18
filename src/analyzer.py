import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class InitialAnalysis(BaseModel):
    summary: str = Field(description="Brief overview: what the project does, what stack it uses, and why it exists (2-3 sentences).")
    options: list[str] = Field(description="Exactly 3 suggested starting points for exploring the codebase (e.g. 'Explore the database layer', 'Trace the main loop', 'Review the API endpoints').")


def get_readme_content(repo_path: str, file_list: list[str]) -> str:
    """
    Finds README.md in the repo and returns its content.
    """
    for file in file_list:
        if file.lower() == "readme.md":
            full_path = os.path.join(repo_path, file)
            with open(full_path, "r", encoding="utf-8") as f:
                # cap at 3000 chars to avoid wasting tokens
                return f.read()[:3000]
    return "README not found."


def generate_first_menu(file_list: list[str], readme_content: str) -> InitialAnalysis:
    """
    Sends project structure to LLM and returns a summary with 3 exploration options.
    """
    # temperature=0.2 keeps responses focused and less creative
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base="https://api.vsellm.ru/v1"
    )

    structured_llm = llm.with_structured_output(InitialAnalysis)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an experienced Senior Developer.
        Your job is to help a Junior developer navigate an unfamiliar repository.
        Analyze the README and file list.
        Return a short summary of the project and suggest 3 logical starting points for exploring the code.
        Be concise and friendly.
        Respond in Russian."""),
        ("human", "README:\n{readme}\n\nFile structure:\n{tree}")
    ])

    chain = prompt | structured_llm

    print("Analyzing project...")

    # cap at 500 files to avoid hitting token limits
    tree_str = "\n".join(file_list[:500])

    return chain.invoke({
        "readme": readme_content,
        "tree": tree_str
    })


# test

if __name__ == "__main__":
    from ingest import clone_repository, build_project_tree

    TEST_URL = "https://github.com/kirill9a/rag-github-tool"

    print("1. Cloning repository...")
    repo_path = clone_repository(TEST_URL)

    print("2. Building file tree...")
    file_list = build_project_tree(repo_path)

    print("3. Reading README...")
    readme = get_readme_content(repo_path, file_list)

    print("\n--- GENERATING START MENU ---")
    result = generate_first_menu(file_list, readme)

    print("\nDone!")
    print("=========================")
    print(f"SUMMARY:\n{result.summary}")
    print("\nOPTIONS:")
    for i, option in enumerate(result.options, 1):
        print(f"  {i}. {option}")
    print("=========================")