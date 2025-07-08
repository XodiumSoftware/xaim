#   Copyright (c) 2025. Xodium.
#   All rights reserved.

import discord


class PollSelect(discord.ui.Select):
    """The select menu for the poll."""

    def __init__(self, options, votes, max_values=1):
        select_options = [discord.SelectOption(label=opt) for opt in options]
        super().__init__(
            placeholder="Cast your vote...",
            options=select_options,
            min_values=1,
            max_values=max_values,
        )
        self.votes = votes

    async def callback(self, interaction: discord.Interaction):
        """Handles a selection from the dropdown."""
        user_id = interaction.user.id

        for voters in self.votes.values():
            if user_id in voters:
                voters.remove(user_id)

        for selected_option in self.values:
            self.votes[selected_option].append(user_id)

        await interaction.response.edit_message(
            embed=self.view.get_embed(), view=self.view
        )
