"""
AIM + MAAP Prompt Wizard

This script walks you through:
- AIM: Actor, Input, Mission
- MAAP: Memory, Assets, Actions, Prompt (style/format)

Then it prints a complete prompt you can copy/paste into ChatGPT or Codex.
"""

def ask_block(title, instructions=None):
    print()
    print(f"=== {title} ===")
    if instructions:
        print(instructions)
    print("(Type your lines. Press ENTER on an empty line to finish.)")

    lines = []
    while True:
        line = input("> ")
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines).strip()

def main():
    print("AIM + MAAP Prompt Wizard")
    print("------------------------")

    # AIM
    actor = ask_block(
        "Actor (A)",
        "Who do you want the AI to be? (e.g., senior Python engineer and baseball data analyst; "
        "or Ernie Bewell, opinionated beat writer.)"
    )

    input_block = ask_block(
        "Input (I)",
        "What are you putting on the table? Describe the situation, data, code, "
        "or context the AI can 'see' right now."
    )

    mission = ask_block(
        "Mission (M)",
        "What do you want done? Be concrete and testable. You can use numbered lines if you like."
    )

    # MAAP
    memory = ask_block(
        "Memory (M)",
        "What relevant history or background should the AI lean on? "
        "(Prior work, preferences, star schema, style, etc.)"
    )

    assets = ask_block(
        "Assets (A)",
        "What concrete resources are in play? (Folders, files, data sets, URLs, etc.)"
    )

    actions = ask_block(
        "Actions (A)",
        "What is the AI allowed/expected to do, and what should it avoid? "
        "(e.g., write Python, propose steps; do NOT invent file names.)"
    )

    prompt_style = ask_block(
        "Prompt (P) – Style & Format",
        "How should the answer feel and what shape should it take? "
        "(e.g., concise, no fluff, step-by-step, plus final code block; "
        "or BBCode with specific sections.)"
    )

    # Build final prompt
    final_prompt = f"""AIM
Actor:
{actor}

Input:
You are given:
{input_block}

Mission:
Your job is to:
{mission}

MAAP
Memory:
Relevant background you should lean on:
{memory}

Assets:
You can assume access to:
{assets}

Actions:
You are allowed/expected to:
{actions}

Prompt (style & format):
{prompt_style}

Now perform the Mission using the AIM + MAAP above.
"""

    print("\n" + "=" * 72)
    print("FINAL PROMPT")
    print("=" * 72)
    print(final_prompt)
    print("=" * 72)
    print("Copy everything between the lines above into ChatGPT or Codex.")
    print()

if __name__ == "__main__":
    main()
