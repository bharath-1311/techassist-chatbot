import asyncio
import websockets



async def communicate_with_server():
    try:
        uri = "ws://localhost:8765"
        async with websockets.connect(uri) as websocket:
            print("Connected to server. Start chatting! Type 'quit' to exit.")
            while True:
                message = input("You: ")
                await websocket.send(message)
                if message.lower() == "quit":
                    break

                response = await websocket.recv()
                print(f"{response}")
    except websockets.ConnectionClosedError as e:
        print(f"connection closed: {e} Retrying...")
        await asyncio.sleep(5)

if __name__ == "__main__":
    while True:
        try:
            asyncio.get_event_loop().run_until_complete(communicate_with_server())
        except Exception as e:
            print(f"Error occurred: {e}. Retrying...")
            asyncio.sleep(5)
