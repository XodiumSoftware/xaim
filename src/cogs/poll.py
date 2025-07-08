#   Copyright (c) 2025. Xodium.
#   All rights reserved.

import discord
from discord.ext import commands

from src.utils import Utils
from src.views.poll import PollView


class Poll(commands.Cog):
    """A cog for Poll-related commands."""

    def __init__(self, bot):
        self.bot = bot
        self.active_polls = []

    @discord.slash_command(
        description="Create a poll with up to 10 options using a dropdown.",
        default_member_permissions=discord.Permissions(administrator=True),
    )
    async def poll(
        self,
        ctx: discord.ApplicationContext,
        question: discord.Option(str, "The question for the poll."),
        options: discord.Option(str, "The poll options, separated by a semicolon (;)."),
        max_choices: discord.Option(
            int,
            "How many options a user can select. Defaults to 1 (single choice).",
            required=False,
            default=1,
        ),
        mention: discord.Option(
            discord.Role,
            "Role to mention with the poll.",
            required=False,
            default=None,
        ),
        deadline: discord.Option(
            str,
            "Poll deadline (e.g., 10s, 30m, 2h, 1d).",
            required=False,
            default=None,
        ),
    ):
        option_list = [opt.strip() for opt in options.split(";") if opt.strip()]

        if len(option_list) < 2:
            await ctx.send_response(
                "Please provide at least two options separated by ';'.", ephemeral=True
            )
            return

        if len(option_list) > 25:
            await ctx.send_response(
                "You can provide a maximum of 25 options.", ephemeral=True
            )
            return

        if max_choices < 1:
            await ctx.send_response(
                "`max_choices` must be 1 or greater.", ephemeral=True
            )
            return

        if max_choices > len(option_list):
            await ctx.send_response(
                "`max_choices` cannot be greater than the number of options.",
                ephemeral=True,
            )
            return

        timeout = Utils.parse_duration(deadline)
        if deadline and timeout is None:
            await ctx.send_response(
                "Invalid deadline format. Use 'm' for minutes, 'h' for hours, 'd' for days (e.g., 30m, 2h, 1d).",
                ephemeral=True,
            )
            return

        view = PollView(question, option_list, ctx.author, max_choices, timeout)
        view.cog = self
        content = mention.mention if mention else None
        await ctx.send_response(content=content, embed=view.get_embed(), view=view)
        view.message = await ctx.interaction.original_response()
        self.active_polls.append(view)


def setup(bot):
    bot.add_cog(Poll(bot))
