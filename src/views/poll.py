#   Copyright (c) 2025. Xodium.
#   All rights reserved.

from datetime import timedelta

import discord

from src.selects.poll import PollSelect


class PollView(discord.ui.View):
    """A view for the poll, containing the voting dropdown."""

    def __init__(self, question, options, author, max_choices=1, timeout=None):
        super().__init__(timeout=timeout)
        self.question = question
        self.options = options
        self.author = author
        self.max_choices = max_choices
        self.votes = {option: [] for option in self.options}
        self.add_item(
            PollSelect(options=self.options, votes=self.votes, max_values=max_choices)
        )
        self.message = None
        self.cog = None
        if timeout:
            self.end_time = discord.utils.utcnow() + timedelta(seconds=timeout)
        else:
            self.end_time = None

    def get_embed(self, closed=False):
        """Creates the poll embed with the current vote counts."""
        description_parts = []
        total_votes = sum(len(voters) for voters in self.votes.values())
        for option, voters in self.votes.items():
            vote_count = len(voters)
            percentage = (
                f"({(vote_count / total_votes) * 100:.1f}%)" if total_votes > 0 else ""
            )
            description_parts.append(f"**{option}**: {vote_count} vote(s) {percentage}")

        description_parts.append("")
        if self.max_choices > 1:
            description_parts.append(f"*Select up to {self.max_choices} options.*")
        else:
            description_parts.append("*Select 1 option.*")

        if self.end_time:
            description_parts.append("")
            if closed:
                timestamp = discord.utils.format_dt(self.end_time, style="f")
                description_parts.append(f"Ended: {timestamp}")
            else:
                timestamp = discord.utils.format_dt(self.end_time, style="R")
                description_parts.append(f"Ends {timestamp}")

        embed = discord.Embed(
            title=f"📊 {self.question}",
            description="\n".join(description_parts),
            color=discord.Colour.blue(),
        )

        footer_text = f"Poll created by {self.author.display_name}"
        embed.set_footer(text=footer_text)
        return embed

    async def on_timeout(self):
        """Disables the view and updates the message when the poll ends."""
        for item in self.children:
            item.disabled = True

        embed = self.get_embed(closed=True)
        embed.title = f"📊 {self.question} (Closed)"
        embed.colour = discord.Colour.red()
        embed.set_footer(text=f"Poll by {self.author.display_name} has ended.")

        if self.message:
            try:
                await self.message.edit(embed=embed, view=self)
            except discord.NotFound:
                pass

        if self.cog and self in self.cog.active_polls:
            self.cog.active_polls.remove(self)
