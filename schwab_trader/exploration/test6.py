#%%
import asyncio
from data.streaming.authenticator import Authenticator
async def main():
    auth = Authenticator()  # Initialize the Authenticator class
    await auth.token_renewal()  # Run the token renewal process

# Check if there is an existing running event loop
if __name__ == "__main__":
    try:
        # If there is an active event loop, use asyncio.create_task
        asyncio.create_task(main())
    except RuntimeError:
        # If no event loop is running, use asyncio.run
        asyncio.run(main())
# %%
