#   Copyright (c) 2025. Xodium.
#   All rights reserved.
import logging
import os
from datetime import datetime

import discord
import dotenv

from src.utils import Utils

dotenv.load_dotenv()


class Bot(discord.AutoShardedBot):
    """Main class for the Discord bot, handling initialization, event setup, and command definitions."""

    LOGGING_PATH = "/data/logs/latest.log"
    if not (TOKEN := os.getenv("TOKEN")):
        raise ValueError(
            "No TOKEN found in environment variables. Please set the TOKEN variable."
        )
    if not (GUILD_ID := os.getenv("GUILD_ID")):
        raise ValueError(
            "No GUILD_ID found in environment variables. Please set the GUILD_ID variable."
        )

    def __init__(self):
        super().__init__(debug_guilds=[int(self.GUILD_ID)])

        self.start_time = datetime.now()
        self.logger = logging.getLogger()

        Utils.setup_logging(self.LOGGING_PATH, 10, logging.INFO)

        self._before_invoke = self.on_before_invoke

        self.load_cogs()

    @property
    def latency_ms(self) -> float:
        """The latency of the bot in milliseconds."""
        return self.latency * 1000

    def load_cogs(self):
        """Dynamically load all cogs from the 'src/cogs' directory."""
        for filename in os.listdir("./src/cogs"):
            if filename.endswith(".py"):
                try:
                    self.load_extension(f"src.cogs.{filename[:-3]}")
                    self.logger.info(f"Loaded cog: {filename}")
                except Exception as e:
                    self.logger.error(f"Failed to load cog {filename}: {e}")

    async def on_ready(self):
        self.logger.info(f"{self.user} (ID={self.user.id}) is ready and online!")

    async def on_before_invoke(self, ctx):
        self.logger.info(
            f"@{ctx.author} (ID={ctx.author.id}) used /{ctx.command.name} in #{ctx.channel} (ID={ctx.channel.id})"
        )

    def run(self, **kwargs):
        super().run(self.TOKEN, **kwargs)


if __name__ == "__main__":
    Bot().run()
