#!/usr/bin/env python3
"""
AIM + MAAP + OCEAN Prompt Builder

This script asks you a series of questions and generates
a structured prompt that follows your AIM + MAAP + OCEAN framework.
"""

def ask_single(prompt: str) -> str:
    """Ask for a single-line answer."""
    print(prompt)
    answer = input("> ").strip()
    return answer

def ask_list(prompt: str) -> list:
    """Ask for a multi-line list. Blank line to finish."""
    print(prompt)
    print("Enter each item on its own line. Leave a blank line to finish.")
    lines = []
    while True:
        line = input("> ").strip()
        if line == "":
            break
        lines.append(line)
    return lines

def format_bullets(lines, prefix="- "):
    """Format a list of lines as bullet points."""
    if not lines:
        return prefix + "(not specified)"
    return "\n".join(f"{prefix}{line}" for line in lines)

def format_numbered(lines):
    """Format a list of lines as numbered items."""
    if not lines:
        return "1) (not specified)"
    return "\n".join(f"{i+1}) {line}" for i, line in enumerate(lines))

def build_prompt(
    actor,
    input_items,
    mission_items,
    memory_items,
    asset_items,
    allowed_actions,
    disallowed_actions,
    extra_style_notes,
    output_format_items,
):
    input_block = format_bullets(input_items)
    mission_block = format_numbered(mission_items)
    memory_block = format_bullets(memory_items)
    assets_block = format_bullets(asset_items)
    allowed_block = format_bullets(allowed_actions)
    disallowed_block = format_bullets(disallowed_actions)
    extra_style_block = format_bullets(extra_style_notes) if extra_style_notes else "- (none specified)"
    output_format_block = format_bullets(output_format_items)

    prompt = f"""AIM
Actor:
You are {actor}.

Input:
You are given:
{input_block}

Mission:
Your job is to:
{mission_block}


MAAP
Memory:
Relevant background you should lean on:
{memory_block}

Assets:
You can assume access to (conceptually):
{assets_block}
If I mention specific file names or columns, use only those;
do not invent new ones.

Actions:
You are allowed/expected to:
{allowed_block}
You should NOT:
{disallowed_block}


Prompt (Style, Format) – OCEAN
Style (OCEAN):
- Original: Avoid generic boilerplate; bring a clear, distinct voice.
- Concrete: Use specific names, examples, file paths, stats, or verses where appropriate.
- Evidence: Support recommendations with brief reasons, logic, or data from the input.
- Assertive: Make clear, confident recommendations ("Do X, then Y") instead of hedging.
- Narrative: Organize the response with a visible arc (setup → development → conclusion),
  so it reads like a coherent story or guided path, not a scattered list.

Additional style notes specific to this task:
{extra_style_block}

Output format:
{output_format_block}

Now perform the Mission using the AIM + MAAP + OCEAN instructions above.
"""
    return prompt

def main():
    print("=== AIM + MAAP + OCEAN Prompt Builder ===\n")

    actor = ask_single("A) Actor – Who do you want the model to be?")

    input_items = ask_list(
        "\nI) Input – What are you giving the model?\n"
        "(Examples: repo paths, data files, scripture passage, scenario constraints)"
    )

    mission_items = ask_list(
        "\nM) Mission – What do you want done?\n"
        "(Enter each mission step/goal on its own line)"
    )

    memory_items = ask_list(
        "\nMAAP: Memory – What prior context or preferences matter here?\n"
        "(Examples: star schema, writing voice, my preferences, previous work)"
    )

    asset_items = ask_list(
        "\nMAAP: Assets – What concrete resources are in play?\n"
        "(Examples: folders, files, links, almanacs, notes)"
    )

    allowed_actions = ask_list(
        "\nMAAP: Actions – What is the model allowed/expected to do?\n"
        "(Examples: write Python, summarize data, draft questions)"
    )

    disallowed_actions = ask_list(
        "\nMAAP: Actions – What should the model NOT do?\n"
        "(Examples: invent columns, change production files, assume libraries I don't have)"
    )

    extra_style_notes = ask_list(
        "\nOCEAN: Any extra style notes beyond OCEAN?\n"
        "(Optional – leave blank if none)"
    )

    output_format_items = ask_list(
        "\nOutput Format – What shape should the answer take?\n"
        "(Examples: bullets + code block, BBCode sections, numbered questions)"
    )

    prompt_text = build_prompt(
        actor,
        input_items,
        mission_items,
        memory_items,
        asset_items,
        allowed_actions,
        disallowed_actions,
        extra_style_notes,
        output_format_items,
    )

    print("\n=== GENERATED PROMPT ===\n")
    print(prompt_text)

    save = ask_single("\nDo you want to save this prompt to a file? (y/n)")
    if save.lower().startswith("y"):
        filename = ask_single("Enter filename (default: prompt.txt)")
        if not filename:
            filename = "prompt.txt"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(prompt_text)
            print(f"Prompt saved to {filename}")
        except Exception as e:
            print(f"Error saving file: {e}")

    print("\nDone. You can now copy/paste this prompt into ChatGPT, Codex, or VS Code.")

if __name__ == "__main__":
    main()
