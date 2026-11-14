import asyncio


async def wait_ready(process, ready):
    async with asyncio.timeout(30):
        while not ready.exists():
            if process.returncode is not None:
                output, _ = await process.communicate()
                raise AssertionError(output.decode())
            await asyncio.sleep(.01)

async def stop_worker(process):
    if process.returncode is None:
        process.kill()
    await process.communicate()
