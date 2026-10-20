from mafia.config import Persona, Personas
from mafia.llm.prompts import template

PLAY_STYLE = {
    "competitive": "You play to win and take losses a bit personally.",
    "win_but_fun": "You want to win, but having fun matters more.",
    "just_for_fun": "You play for laughs and chaos more than for winning.",
    "hang_out": "You're mostly here to hang out with friends; the game is secondary.",
}
UPDATE_STYLE = {
    "stubborn": "You change your mind slowly. One game rarely convinces you; you need to see something repeatedly.",
    "impulsive": "You change your mind fast, sometimes after a single moment.",
    "paranoid": "You tend to read hidden meaning into small things.",
    "insecure": "You often doubt yourself and sometimes overcorrect.",
    "balanced": "You weigh the evidence and change your mind when it adds up.",
}
SELF_INSIGHT = {
    "high": "You usually know how you come across.",
    "medium": "You sometimes misjudge how you come across.",
    "low": "You often misjudge how you come across to others.",
}
NEUTRAL_SELF = "I'm still figuring out how I come across in this group."


def friends(p: Persona, ps: Personas) -> list[Persona]:
    """The six other players, in seat order."""
    return sorted((o for o in ps.personas if o.agent_id != p.agent_id), key=lambda o: o.seat)


def persona_block(p: Persona, ps: Personas) -> str:
    return template("persona").format(
        name=p.name, age=p.age, background=p.background, voice=p.voice,
        example_1=p.examples[0], example_2=p.examples[1], how_you_play=p.how_you_play,
        play_style_line=PLAY_STYLE[p.play_style], update_style_line=UPDATE_STYLE[p.update_style],
        self_insight_line=SELF_INSIGHT[p.self_insight],
        friend_names=", ".join(f.name for f in friends(p, ps)),
    )


def starting_impressions(p: Persona, ps: Personas) -> dict[str, str]:
    """agent_id -> personality-only impression at game 1, plus the private line for the closest friend."""
    out = {}
    for f in friends(p, ps):
        out[f.agent_id] = ps.group_impressions[f.agent_id]
        if f.agent_id == p.closest_friend and p.private_line_about_closest:
            out[f.agent_id] += " " + p.private_line_about_closest
    return out


def friend_lines(p: Persona, ps: Personas, by_id: dict[str, str]) -> str:
    return "\n".join(f"- {f.name}: {by_id[f.agent_id]}" for f in friends(p, ps))


def starting_summaries(p: Persona, ps: Personas) -> tuple[str, str]:
    """(friend_summaries text, self summary) at game 1."""
    return friend_lines(p, ps, starting_impressions(p, ps)), NEUTRAL_SELF
