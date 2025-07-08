#   Copyright (c) 2025. Xodium.
#   All rights reserved.

import platform
import sys
from datetime import datetime

import discord
import psutil
from discord.ext import commands

from src.utils import Utils


class Metrics(commands.Cog):
    """A cog for handling Metrics-related commands."""

    def __init__(self, bot):
        self.bot = bot

    @discord.slash_command(
        description="Displays the bot's metrics.",
        default_member_permissions=discord.Permissions(administrator=True),
    )
    async def metrics(self, ctx: discord.ApplicationContext):
        mem = psutil.virtual_memory()
        mem_used = mem.used / (1024**3)
        mem_total = mem.total / (1024**3)

        await ctx.send_response(
            embed=discord.Embed(
                title="📈 Metrics",
                description=(
                    f"**Performance**\n"
                    f"Latency: `{self.bot.latency * 1000:.2f} ms`\n"
                    f"Uptime: `{Utils.format_uptime(datetime.now() - self.bot.start_time)}`\n\n"
                    f"**Bot**\n"
                    f"Commands: `{len(self.bot.commands)}`\n"
                    f"Cogs: `{len(self.bot.cogs)}`\n\n"
                    f"**System**\n"
                    f"CPU Usage: `{psutil.cpu_percent()}%`\n"
                    f"Memory Usage: `{mem_used:.2f} GB / {mem_total:.2f} GB ({mem.percent}%)`\n"
                    f"Python Version: `{sys.version.split(' ')[0]}`\n"
                    f"Operating System: `{platform.system()} {platform.release()}`\n\n"
                    f"**Statistics**\n"
                    f"Guilds: `{len(self.bot.guilds)}`\n"
                    f"Users: `{len(self.bot.users)}`\n"
                    f"discord.py: `v{discord.__version__}`"
                ),
                color=discord.Colour.blue(),
            ),
            ephemeral=True,
        )


def setup(bot):
    bot.add_cog(Metrics(bot))
